%%% Test helper: synthesize GGUF binaries in-memory so the parser, tokenizer
%%% and forward pass can be exercised without downloading a real model.
-module(ar_synth).

-export([build/2, meta_str/2, meta_u32/2, meta_f32/2, meta_arr_str/2,
         meta_arr_i32/2, f32_tensor/3, raw_tensor/4,
         tiny_llama_gguf/0, gpt2_tokenizer_meta/0]).

-define(ALIGN, 32).

%% ---------------------------------------------------------------------------
%% Low-level GGUF serialization
%% ---------------------------------------------------------------------------

%% MetaList  :: [{KeyBin, TypeCode, ValBin}]
%% TensorList:: [{NameBin, Dims, TypeCode, DataBin}]
build(MetaList, TensorList) ->
    KvBin = iolist_to_binary([enc_kv(K, T, V) || {K, T, V} <- MetaList]),
    %% Assign 32-aligned offsets within the tensor data section.
    {Infos, DataChunks, _} =
        lists:foldl(
          fun({Name, Dims, Type, Data}, {IAcc, DAcc, Cursor}) ->
              Off = align_up(Cursor, ?ALIGN),
              Pad = Off - Cursor,
              Info = enc_tensor_info(Name, Dims, Type, Off),
              {[Info | IAcc],
               [<<0:(Pad * 8), Data/binary>> | DAcc],
               Off + byte_size(Data)}
          end, {[], [], 0}, TensorList),
    InfoBin = iolist_to_binary(lists:reverse(Infos)),
    Header = <<16#47, 16#47, 16#55, 16#46,          %% "GGUF"
               3:32/little,                          %% version
               (length(TensorList)):64/little,
               (length(MetaList)):64/little>>,
    Pre = <<Header/binary, KvBin/binary, InfoBin/binary>>,
    PrePad = align_up(byte_size(Pre), ?ALIGN) - byte_size(Pre),
    DataBin = iolist_to_binary(lists:reverse(DataChunks)),
    <<Pre/binary, 0:(PrePad * 8), DataBin/binary>>.

enc_str(B) -> <<(byte_size(B)):64/little, B/binary>>.

enc_kv(Key, Type, Val) -> <<(enc_str(Key))/binary, Type:32/little, Val/binary>>.

enc_tensor_info(Name, Dims, Type, Off) ->
    DimsBin = iolist_to_binary([<<D:64/little>> || D <- Dims]),
    <<(enc_str(Name))/binary, (length(Dims)):32/little, DimsBin/binary,
      Type:32/little, Off:64/little>>.

%% ---------------------------------------------------------------------------
%% Metadata value builders (return {KeyBin, TypeCode, ValBin})
%% ---------------------------------------------------------------------------

meta_str(Key, S) -> {Key, 8, enc_str(S)}.
meta_u32(Key, V) -> {Key, 4, <<V:32/little>>}.
meta_f32(Key, V) -> {Key, 6, <<V:32/little-float>>}.

meta_arr_str(Key, Strs) ->
    Body = iolist_to_binary([enc_str(S) || S <- Strs]),
    {Key, 9, <<8:32/little, (length(Strs)):64/little, Body/binary>>}.

meta_arr_i32(Key, Ints) ->
    Body = iolist_to_binary([<<I:32/little-signed>> || I <- Ints]),
    {Key, 9, <<5:32/little, (length(Ints)):64/little, Body/binary>>}.

%% ---------------------------------------------------------------------------
%% Tensor builders
%% ---------------------------------------------------------------------------

%% F32 tensor from a flat list of floats. Dims are GGUF ne = [ne0, ne1, ...].
f32_tensor(Name, Dims, Floats) ->
    Data = iolist_to_binary([<<F:32/little-float>> || F <- Floats]),
    {Name, Dims, 0, Data}.

%% Raw tensor: caller supplies the exact data bytes and ggml type code.
raw_tensor(Name, Dims, Type, Data) -> {Name, Dims, Type, Data}.

%% ---------------------------------------------------------------------------
%% A complete tiny llama model (all F32 weights) + embedded gpt2 tokenizer.
%% ---------------------------------------------------------------------------

tiny_llama_gguf() ->
    Vocab = 6, Embd = 8, NHead = 2, NLayer = 2, Ffn = 16,
    HeadDim = Embd div NHead,
    Meta = [meta_str(<<"general.architecture">>, <<"llama">>),
            meta_str(<<"general.name">>, <<"tiny-llama">>),
            meta_u32(<<"llama.attention.head_count">>, NHead),
            meta_u32(<<"llama.attention.head_count_kv">>, NHead),
            meta_u32(<<"llama.block_count">>, NLayer),
            meta_u32(<<"llama.embedding_length">>, Embd),
            meta_u32(<<"llama.feed_forward_length">>, Ffn),
            meta_u32(<<"llama.rope.dimension_count">>, HeadDim),
            meta_f32(<<"llama.attention.layer_norm_rms_epsilon">>, 1.0e-5),
            meta_u32(<<"llama.context_length">>, 64)
            | gpt2_tokenizer_meta()],
    %% Weight tensors: dims are [in, out] (ne0=in fastest), data is out rows of in.
    Mat = fun(Name, Out, In) -> f32_tensor(Name, [In, Out], det(Out * In)) end,
    Vec = fun(Name, N) -> f32_tensor(Name, [N], ones(N)) end,
    Layers = lists:append(
               [[Vec(p(I, "attn_norm.weight"), Embd),
                 Vec(p(I, "ffn_norm.weight"), Embd),
                 Mat(p(I, "attn_q.weight"), Embd, Embd),
                 Mat(p(I, "attn_k.weight"), Embd, Embd),
                 Mat(p(I, "attn_v.weight"), Embd, Embd),
                 Mat(p(I, "attn_output.weight"), Embd, Embd),
                 Mat(p(I, "ffn_gate.weight"), Ffn, Embd),
                 Mat(p(I, "ffn_up.weight"), Ffn, Embd),
                 Mat(p(I, "ffn_down.weight"), Embd, Ffn)]
                || I <- lists:seq(0, NLayer - 1)]),
    Tensors = [f32_tensor(<<"token_embd.weight">>, [Embd, Vocab], det(Vocab * Embd)),
               Vec(<<"output_norm.weight">>, Embd),
               Mat(<<"output.weight">>, Vocab, Embd)
               | Layers],
    build(Meta, Tensors).

%% Embedded gpt2 tokenizer metadata shared by tests. Single-char base tokens
%% plus a merge so encoding "ab" -> id 4.
gpt2_tokenizer_meta() ->
    [meta_str(<<"tokenizer.ggml.model">>, <<"gpt2">>),
     meta_arr_str(<<"tokenizer.ggml.tokens">>,
                  [<<"<bos>">>, <<"a">>, <<"b">>, <<"c">>, <<"ab">>, <<"<eos>">>]),
     meta_arr_str(<<"tokenizer.ggml.merges">>, [<<"a b">>]),
     meta_arr_i32(<<"tokenizer.ggml.token_type">>, [3, 1, 1, 1, 1, 3]),
     meta_u32(<<"tokenizer.ggml.bos_token_id">>, 0),
     meta_u32(<<"tokenizer.ggml.eos_token_id">>, 5)].

%% ---------------------------------------------------------------------------
%% helpers
%% ---------------------------------------------------------------------------

p(I, Suffix) ->
    <<"blk.", (integer_to_binary(I))/binary, ".", (list_to_binary(Suffix))/binary>>.

%% Small deterministic weights (finite, varied) for a stable forward pass.
det(N) -> [0.02 * (((X * 7 + 3) rem 11) - 5) || X <- lists:seq(0, N - 1)].
ones(N) -> lists:duplicate(N, 1.0).

align_up(N, A) -> ((N + A - 1) div A) * A.
