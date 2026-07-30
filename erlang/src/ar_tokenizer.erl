%%% GPT-2 style byte-level BPE tokenizer, reconstructed from the
%%% `tokenizer.ggml.*` arrays embedded in a GGUF file. Covers the "gpt2"
%%% tokenizer family used by Llama 3, Qwen2, Phi-3, GPT-2, etc.
%%%
%%% (SentencePiece/"llama" unigram models are not handled here; pass a real
%%% tokenizer or use a gpt2-family GGUF.)
-module(ar_tokenizer).

-export([from_gguf/1, encode/2, encode/3, decode/2, decode/3, bos_id/1, eos_id/1]).

-record(tok, {tokens :: tuple(),          %% id -> token binary (byte-level)
              vocab :: map(),             %% token binary -> id
              merges :: map(),            %% {A,B} -> rank
              specials :: sets:set(),     %% set of special ids
              bos :: integer() | undefined,
              eos :: integer() | undefined,
              encoder :: map(),           %% byte (0..255) -> codepoint
              decoder :: map(),           %% codepoint -> byte
              re :: term()}).

-opaque tokenizer() :: #tok{}.
-export_type([tokenizer/0]).

%% ---------------------------------------------------------------------------
%% Construction
%% ---------------------------------------------------------------------------

-spec from_gguf(ar_gguf:gguf()) -> tokenizer().
from_gguf(G) ->
    case ar_gguf:meta(G, <<"tokenizer.ggml.model">>) of
        <<"gpt2">> -> ok;
        Other -> erlang:error({unsupported_tokenizer, Other})
    end,
    Tokens = ar_gguf:meta(G, <<"tokenizer.ggml.tokens">>),
    MergesRaw = ar_gguf:meta(G, <<"tokenizer.ggml.merges">>, []),
    Types = ar_gguf:meta(G, <<"tokenizer.ggml.token_type">>, []),
    Bos = ar_gguf:meta(G, <<"tokenizer.ggml.bos_token_id">>),
    Eos = ar_gguf:meta(G, <<"tokenizer.ggml.eos_token_id">>),

    Vocab = maps:from_list(
              [{T, I} || {T, I} <- lists:zip(Tokens, lists:seq(0, length(Tokens) - 1))]),
    Merges = build_merges(MergesRaw),
    Specials = build_specials(Types, Bos, Eos),
    {Enc, Dec} = build_byte_maps(),
    RE = compile_pattern(),
    #tok{tokens = list_to_tuple(Tokens), vocab = Vocab, merges = Merges,
         specials = Specials, bos = Bos, eos = Eos,
         encoder = Enc, decoder = Dec, re = RE}.

build_merges(MergesRaw) ->
    Indexed = lists:zip(MergesRaw, lists:seq(0, length(MergesRaw) - 1)),
    lists:foldl(fun({M, Rank}, Acc) ->
                    case binary:split(M, <<" ">>) of
                        [A, B] -> Acc#{{A, B} => Rank};
                        _ -> Acc
                    end
                end, #{}, Indexed).

build_specials(Types, Bos, Eos) ->
    FromTypes = case Types of
                    [] -> [];
                    _ -> [I || {Ty, I} <- lists:zip(Types, lists:seq(0, length(Types) - 1)),
                               Ty =:= 3]   %% 3 = CONTROL
                end,
    Extra = [X || X <- [Bos, Eos], X =/= undefined],
    sets:from_list(FromTypes ++ Extra).

bos_id(#tok{bos = B}) -> B.
eos_id(#tok{eos = E}) -> E.

%% ---------------------------------------------------------------------------
%% Encoding
%% ---------------------------------------------------------------------------

-spec encode(tokenizer(), binary() | string()) -> [integer()].
encode(Tok, Text) -> encode(Tok, Text, false).

%% AddBos: prepend the BOS token id if known.
-spec encode(tokenizer(), binary() | string(), boolean()) -> [integer()].
encode(Tok, Text, AddBos) when is_list(Text) ->
    encode(Tok, unicode:characters_to_binary(Text), AddBos);
encode(#tok{re = RE} = Tok, Text, AddBos) when is_binary(Text) ->
    Chunks = pretokenize(RE, Text),
    Ids = lists:append([encode_chunk(Tok, C) || C <- Chunks]),
    case AddBos andalso Tok#tok.bos =/= undefined of
        true -> [Tok#tok.bos | Ids];
        false -> Ids
    end.

pretokenize(nomatch, Text) -> [Text];
pretokenize(RE, Text) ->
    case re:run(Text, RE, [global, {capture, all, binary}]) of
        {match, Matches} -> [M || [M | _] <- Matches];
        nomatch -> []
    end.

encode_chunk(#tok{encoder = Enc} = Tok, Chunk) ->
    Bytes = binary_to_list(Chunk),
    Symbols = [unicode:characters_to_binary([maps:get(B, Enc)]) || B <- Bytes],
    Merged = bpe(Symbols, Tok#tok.merges),
    [lookup(Tok, S) || S <- Merged].

lookup(#tok{vocab = V}, Sym) ->
    case maps:get(Sym, V, undefined) of
        undefined -> erlang:error({token_not_in_vocab, Sym});
        Id -> Id
    end.

%% Byte-pair merging: repeatedly merge the adjacent pair with the lowest rank.
bpe([], _Merges) -> [];
bpe([_] = Syms, _Merges) -> Syms;
bpe(Syms, Merges) ->
    case best_pair(Syms, Merges) of
        none -> Syms;
        Pair -> bpe(merge_pair(Syms, Pair), Merges)
    end.

best_pair(Syms, Merges) ->
    Pairs = adjacent_pairs(Syms),
    Ranked = [{maps:get(P, Merges, none), P} || P <- Pairs],
    Present = [{R, P} || {R, P} <- Ranked, R =/= none],
    case Present of
        [] -> none;
        _ -> {_R, P} = lists:min(Present), P
    end.

adjacent_pairs([A, B | T]) -> [{A, B} | adjacent_pairs([B | T])];
adjacent_pairs(_) -> [].

merge_pair([A, B | T], {A, B}) ->
    [<<A/binary, B/binary>> | merge_pair(T, {A, B})];
merge_pair([X | T], Pair) ->
    [X | merge_pair(T, Pair)];
merge_pair([], _Pair) ->
    [].

%% ---------------------------------------------------------------------------
%% Decoding
%% ---------------------------------------------------------------------------

-spec decode(tokenizer(), [integer()]) -> binary().
decode(Tok, Ids) -> decode(Tok, Ids, true).

%% SkipSpecial: drop special tokens (BOS/EOS/control) from the output.
-spec decode(tokenizer(), [integer()], boolean()) -> binary().
decode(#tok{tokens = Toks, decoder = Dec, specials = Sp}, Ids, SkipSpecial) ->
    Keep = case SkipSpecial of
               true -> [Id || Id <- Ids, not sets:is_element(Id, Sp)];
               false -> Ids
           end,
    Pieces = [element(Id + 1, Toks) || Id <- Keep],
    Concat = iolist_to_binary(Pieces),
    Codepoints = unicode:characters_to_list(Concat, utf8),
    Bytes = [maps:get(Cp, Dec, $?) || Cp <- Codepoints],
    list_to_binary(Bytes).

%% ---------------------------------------------------------------------------
%% GPT-2 byte<->unicode maps
%% ---------------------------------------------------------------------------

build_byte_maps() ->
    Printable = lists:seq(16#21, 16#7E) ++ lists:seq(16#A1, 16#AC) ++ lists:seq(16#AE, 16#FF),
    PrintableSet = sets:from_list(Printable),
    %% Bytes not in the printable set get remapped to codepoints 256, 257, ...
    {Enc, _} = lists:foldl(
                 fun(B, {Map, N}) ->
                     case sets:is_element(B, PrintableSet) of
                         true -> {Map#{B => B}, N};
                         false -> {Map#{B => 256 + N}, N + 1}
                     end
                 end, {#{}, 0}, lists:seq(0, 255)),
    Dec = maps:fold(fun(Byte, Cp, Acc) -> Acc#{Cp => Byte} end, #{}, Enc),
    {Enc, Dec}.

compile_pattern() ->
    Pat = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    case re:compile(Pat, [unicode]) of
        {ok, RE} -> RE;
        {error, _} -> nomatch
    end.
