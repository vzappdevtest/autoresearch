%%% GGUF file parser (pure Erlang, using binary bit-syntax).
%%%
%%% Parses the GGUF container used by llama.cpp / GGML: header, metadata
%%% key/value table, and the tensor info table. Tensor *data* is left in the
%%% original binary and read/dequantized lazily via tensor_f32/2.
%%%
%%% Supported tensor dtypes for dequantization: F32, F16, Q8_0, Q4_0.
%%% (Any GGUF can still be inspected; only these can be turned into floats.)
-module(ar_gguf).

-export([read_file/1, parse/1, architecture/1, meta/2, meta/3, metadata_keys/1,
         tensor_names/1, tensor_info/2, tensor_f32/2, tensor_rows/2,
         value_to_string/1, dtype_name/1]).

%% ggml dtype ids we handle explicitly.
-define(GGML_F32,  0).
-define(GGML_F16,  1).
-define(GGML_Q4_0, 2).
-define(GGML_Q8_0, 8).

-define(QK, 32). %% block size for Q4_0 / Q8_0

%% A parsed GGUF file.
-record(gguf, {version :: integer(),
               metadata :: map(),               %% key(binary) => term()
               tensors :: map(),                %% name(binary) => {Dims, Type, Offset}
               data_offset :: integer(),        %% byte offset of tensor data section
               bin :: binary()}).                %% whole file

-opaque gguf() :: #gguf{}.
-export_type([gguf/0]).

%% ---------------------------------------------------------------------------
%% Public API
%% ---------------------------------------------------------------------------

-spec read_file(string()) -> {ok, gguf()} | {error, term()}.
read_file(Path) ->
    case file:read_file(Path) of
        {ok, Bin} -> {ok, parse(Bin)};
        {error, Reason} -> {error, Reason}
    end.

-spec parse(binary()) -> gguf().
parse(Bin) ->
    <<16#47, 16#47, 16#55, 16#46,           %% "GGUF" magic (G G U F)
      Version:32/little,
      TensorCount:64/little,
      KvCount:64/little,
      Rest0/binary>> = Bin,
    {Meta, Rest1} = parse_kv(KvCount, Rest0, #{}),
    {Tensors, Rest2} = parse_tensor_infos(TensorCount, Rest1, #{}),
    Consumed = byte_size(Bin) - byte_size(Rest2),
    Align = maps:get(<<"general.alignment">>, Meta, 32),
    DataOffset = align_up(Consumed, Align),
    #gguf{version = Version, metadata = Meta, tensors = Tensors,
          data_offset = DataOffset, bin = Bin}.

-spec architecture(gguf()) -> binary() | undefined.
architecture(G) -> meta(G, <<"general.architecture">>, undefined).

-spec meta(gguf(), binary()) -> term() | undefined.
meta(G, Key) -> meta(G, Key, undefined).

-spec meta(gguf(), binary(), term()) -> term().
meta(#gguf{metadata = M}, Key, Default) -> maps:get(Key, M, Default).

-spec metadata_keys(gguf()) -> [binary()].
metadata_keys(#gguf{metadata = M}) -> maps:keys(M).

-spec tensor_names(gguf()) -> [binary()].
tensor_names(#gguf{tensors = T}) -> maps:keys(T).

%% {Dims, Type, Offset} for a tensor, or undefined.
tensor_info(#gguf{tensors = T}, Name) -> maps:get(Name, T, undefined).

%% Return the tensor's data as a flat list of floats (row-major).
-spec tensor_f32(gguf(), binary()) -> [float()].
tensor_f32(#gguf{tensors = T, data_offset = Off, bin = Bin}, Name) ->
    case maps:get(Name, T, undefined) of
        undefined -> erlang:error({no_such_tensor, Name});
        {Dims, Type, TOff} ->
            NElem = lists:foldl(fun(D, A) -> D * A end, 1, Dims),
            Start = Off + TOff,
            NBytes = nbytes(Type, NElem),
            <<_:Start/binary, Raw:NBytes/binary, _/binary>> = Bin,
            dequantize(Type, Raw, NElem)
    end.

%% Return a 2D tensor as a list of rows. GGUF stores dims as [ne0, ne1, ...]
%% where ne0 is the fastest-moving axis (row length). So rows = ne1, cols = ne0.
-spec tensor_rows(gguf(), binary()) -> [[float()]].
tensor_rows(G, Name) ->
    {Dims, _Type, _Off} = tensor_info(G, Name),
    Flat = tensor_f32(G, Name),
    case Dims of
        [Cols, Rows | _] -> chunk(Flat, Cols, Rows);
        [Cols] -> chunk(Flat, Cols, 1)
    end.

%% ---------------------------------------------------------------------------
%% Metadata parsing
%% ---------------------------------------------------------------------------

parse_kv(0, Rest, Acc) -> {Acc, Rest};
parse_kv(N, Bin, Acc) ->
    {Key, Rest0} = parse_string(Bin),
    <<ValType:32/little, Rest1/binary>> = Rest0,
    {Value, Rest2} = parse_value(ValType, Rest1),
    parse_kv(N - 1, Rest2, Acc#{Key => Value}).

parse_string(<<Len:64/little, Str:Len/binary, Rest/binary>>) -> {Str, Rest}.

parse_value(0, <<V:8/unsigned, R/binary>>) -> {V, R};                   %% uint8
parse_value(1, <<V:8/signed, R/binary>>) -> {V, R};                     %% int8
parse_value(2, <<V:16/little-unsigned, R/binary>>) -> {V, R};           %% uint16
parse_value(3, <<V:16/little-signed, R/binary>>) -> {V, R};             %% int16
parse_value(4, <<V:32/little-unsigned, R/binary>>) -> {V, R};           %% uint32
parse_value(5, <<V:32/little-signed, R/binary>>) -> {V, R};             %% int32
parse_value(6, <<V:32/little-float, R/binary>>) -> {V, R};              %% float32
parse_value(7, <<V:8, R/binary>>) -> {V =/= 0, R};                      %% bool
parse_value(8, Bin) -> parse_string(Bin);                               %% string
parse_value(10, <<V:64/little-unsigned, R/binary>>) -> {V, R};          %% uint64
parse_value(11, <<V:64/little-signed, R/binary>>) -> {V, R};            %% int64
parse_value(12, <<V:64/little-float, R/binary>>) -> {V, R};             %% float64
parse_value(9, <<ElemType:32/little, Count:64/little, R/binary>>) ->    %% array
    parse_array(ElemType, Count, R, []).

parse_array(_ElemType, 0, Rest, Acc) -> {lists:reverse(Acc), Rest};
parse_array(ElemType, N, Bin, Acc) ->
    {V, Rest} = parse_value(ElemType, Bin),
    parse_array(ElemType, N - 1, Rest, [V | Acc]).

%% ---------------------------------------------------------------------------
%% Tensor info parsing
%% ---------------------------------------------------------------------------

parse_tensor_infos(0, Rest, Acc) -> {Acc, Rest};
parse_tensor_infos(N, Bin, Acc) ->
    {Name, Rest0} = parse_string(Bin),
    <<NDims:32/little, Rest1/binary>> = Rest0,
    {Dims, Rest2} = parse_dims(NDims, Rest1, []),
    <<Type:32/little, Offset:64/little, Rest3/binary>> = Rest2,
    parse_tensor_infos(N - 1, Rest3, Acc#{Name => {Dims, Type, Offset}}).

parse_dims(0, Rest, Acc) -> {lists:reverse(Acc), Rest};
parse_dims(N, <<D:64/little, Rest/binary>>, Acc) ->
    parse_dims(N - 1, Rest, [D | Acc]).

%% ---------------------------------------------------------------------------
%% Dequantization
%% ---------------------------------------------------------------------------

nbytes(?GGML_F32, N) -> N * 4;
nbytes(?GGML_F16, N) -> N * 2;
nbytes(?GGML_Q8_0, N) -> (N div ?QK) * 34;
nbytes(?GGML_Q4_0, N) -> (N div ?QK) * 18;
nbytes(Type, _N) -> erlang:error({unsupported_dtype_for_dequant, dtype_name(Type)}).

dequantize(?GGML_F32, Raw, _N) ->
    [V || <<V:32/little-float>> <= Raw];
dequantize(?GGML_F16, Raw, _N) ->
    [half_to_float(H) || <<H:16/little-unsigned>> <= Raw];
dequantize(?GGML_Q8_0, Raw, _N) ->
    deq_q8_0(Raw, []);
dequantize(?GGML_Q4_0, Raw, _N) ->
    deq_q4_0(Raw, []);
dequantize(Type, _Raw, _N) ->
    erlang:error({unsupported_dtype_for_dequant, dtype_name(Type)}).

%% Q8_0 block: fp16 scale + 32 int8 quants (34 bytes).
deq_q8_0(<<>>, Acc) -> lists:reverse(Acc);
deq_q8_0(<<D:16/little-unsigned, Qs:32/binary, Rest/binary>>, Acc) ->
    Scale = half_to_float(D),
    Vals = [Scale * Q || <<Q:8/signed>> <= Qs],
    deq_q8_0(Rest, lists:reverse(Vals, Acc)).

%% Q4_0 block: fp16 scale + 16 bytes of packed 4-bit quants (18 bytes).
%% Element i uses the low nibble; element i+16 uses the high nibble.
deq_q4_0(<<>>, Acc) -> lists:reverse(Acc);
deq_q4_0(<<D:16/little-unsigned, Qs:16/binary, Rest/binary>>, Acc) ->
    Scale = half_to_float(D),
    Bytes = binary_to_list(Qs),
    Lows = [Scale * ((B band 16#0F) - 8) || B <- Bytes],
    Highs = [Scale * ((B bsr 4) - 8) || B <- Bytes],
    Block = Lows ++ Highs,
    deq_q4_0(Rest, lists:reverse(Block, Acc)).

%% IEEE-754 half precision -> Erlang float.
half_to_float(H) ->
    Sign = (H bsr 15) band 1,
    Exp = (H bsr 10) band 16#1F,
    Mant = H band 16#3FF,
    S = case Sign of 0 -> 1.0; _ -> -1.0 end,
    if
        Exp =:= 0 andalso Mant =:= 0 -> S * 0.0;
        Exp =:= 0 -> S * math:pow(2, -14) * (Mant / 1024.0);      %% subnormal
        Exp =:= 16#1F -> S * 1.0e38;                              %% inf/nan approx
        true -> S * math:pow(2, Exp - 15) * (1.0 + Mant / 1024.0)
    end.

%% ---------------------------------------------------------------------------
%% Display helpers
%% ---------------------------------------------------------------------------

value_to_string(V) when is_integer(V) -> integer_to_list(V);
value_to_string(V) when is_float(V) -> io_lib:format("~g", [V]);
value_to_string(true) -> "true";
value_to_string(false) -> "false";
value_to_string(V) when is_binary(V) -> truncate(binary_to_list(V), 80);
value_to_string(V) when is_list(V) ->
    N = length(V),
    Preview = [value_to_string(X) || X <- lists:sublist(V, 6)],
    Joined = string:join(Preview, ", "),
    if N > 6 -> io_lib:format("[~s, ... ~p items]", [Joined, N]);
       true -> io_lib:format("[~s]", [Joined])
    end.

truncate(S, Max) ->
    case length(S) > Max of
        true -> lists:sublist(S, Max) ++ "...";
        false -> S
    end.

dtype_name(0) -> "F32";
dtype_name(1) -> "F16";
dtype_name(2) -> "Q4_0";
dtype_name(3) -> "Q4_1";
dtype_name(6) -> "Q5_0";
dtype_name(7) -> "Q5_1";
dtype_name(8) -> "Q8_0";
dtype_name(9) -> "Q8_1";
dtype_name(10) -> "Q2_K";
dtype_name(11) -> "Q3_K";
dtype_name(12) -> "Q4_K";
dtype_name(13) -> "Q5_K";
dtype_name(14) -> "Q6_K";
dtype_name(15) -> "Q8_K";
dtype_name(N) -> "type" ++ integer_to_list(N).

%% ---------------------------------------------------------------------------
%% Utilities
%% ---------------------------------------------------------------------------

align_up(N, A) -> ((N + A - 1) div A) * A.

chunk(_List, _Cols, 0) -> [];
chunk(List, Cols, Rows) ->
    {Row, Rest} = lists:split(Cols, List),
    [Row | chunk(Rest, Cols, Rows - 1)].
