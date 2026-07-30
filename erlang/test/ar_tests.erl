%%% EUnit tests for the pure-Erlang GGUF LLM engine. All fixtures are
%%% synthesized in-memory (see ar_synth) — no model downloads required.
-module(ar_tests).

-include_lib("eunit/include/eunit.hrl").

%% ---------------------------------------------------------------------------
%% GGUF parsing
%% ---------------------------------------------------------------------------

parse_header_test() ->
    G = ar_gguf:parse(ar_synth:tiny_llama_gguf()),
    ?assertEqual(<<"llama">>, ar_gguf:architecture(G)),
    ?assertEqual(8, ar_gguf:meta(G, <<"llama.embedding_length">>)),
    ?assertEqual(2, ar_gguf:meta(G, <<"llama.block_count">>)),
    %% token_embd.weight present with the right shape.
    ?assert(lists:member(<<"token_embd.weight">>, ar_gguf:tensor_names(G))).

%% ---------------------------------------------------------------------------
%% Dequantization against hand-crafted blocks
%% ---------------------------------------------------------------------------

dequant_f16_test() ->
    %% halves for [1.0, 2.0, 0.5, -2.0], little-endian.
    Data = <<16#00, 16#3C, 16#00, 16#40, 16#00, 16#38, 16#00, 16#C0>>,
    Meta = [ar_synth:meta_str(<<"general.architecture">>, <<"x">>)],
    Bin = ar_synth:build(Meta, [ar_synth:raw_tensor(<<"t">>, [4], 1, Data)]),
    G = ar_gguf:parse(Bin),
    ?assertEqual([1.0, 2.0, 0.5, -2.0], ar_gguf:tensor_f32(G, <<"t">>)).

dequant_q8_0_test() ->
    %% scale 0.5 (half 0x3800) + int8 quants 1..32 -> 0.5, 1.0, ..., 16.0
    Qs = list_to_binary(lists:seq(1, 32)),
    Data = <<16#00, 16#38, Qs/binary>>,
    Meta = [ar_synth:meta_str(<<"general.architecture">>, <<"x">>)],
    Bin = ar_synth:build(Meta, [ar_synth:raw_tensor(<<"t">>, [32], 8, Data)]),
    G = ar_gguf:parse(Bin),
    Out = ar_gguf:tensor_f32(G, <<"t">>),
    ?assertEqual(32, length(Out)),
    ?assertEqual(0.5, hd(Out)),
    ?assertEqual(16.0, lists:last(Out)).

dequant_q4_0_test() ->
    %% scale 0.5, all bytes 0x80 -> low nibble 0 -> (0-8)*0.5 = -4.0 (first 16),
    %% high nibble 8 -> (8-8)*0.5 = 0.0 (next 16).
    Qs = binary:copy(<<16#80>>, 16),
    Data = <<16#00, 16#38, Qs/binary>>,
    Meta = [ar_synth:meta_str(<<"general.architecture">>, <<"x">>)],
    Bin = ar_synth:build(Meta, [ar_synth:raw_tensor(<<"t">>, [32], 2, Data)]),
    G = ar_gguf:parse(Bin),
    Out = ar_gguf:tensor_f32(G, <<"t">>),
    ?assertEqual(lists:duplicate(16, -4.0) ++ lists:duplicate(16, 0.0), Out).

%% ---------------------------------------------------------------------------
%% Tokenizer
%% ---------------------------------------------------------------------------

tokenizer_roundtrip_test() ->
    Meta = [ar_synth:meta_str(<<"general.architecture">>, <<"llama">>)
            | ar_synth:gpt2_tokenizer_meta()],
    G = ar_gguf:parse(ar_synth:build(Meta, [])),
    Tok = ar_tokenizer:from_gguf(G),
    ?assertEqual(0, ar_tokenizer:bos_id(Tok)),
    ?assertEqual(5, ar_tokenizer:eos_id(Tok)),
    %% "ab" merges to the single token id 4.
    ?assertEqual([4], ar_tokenizer:encode(Tok, <<"ab">>)),
    %% with BOS prepended.
    ?assertEqual([0, 4], ar_tokenizer:encode(Tok, <<"ab">>, true)),
    %% decode roundtrip (special-free).
    ?assertEqual(<<"ab">>, ar_tokenizer:decode(Tok, [4])),
    %% specials are skipped by default on decode.
    ?assertEqual(<<"ab">>, ar_tokenizer:decode(Tok, [0, 4, 5], true)).

%% ---------------------------------------------------------------------------
%% Tensor math
%% ---------------------------------------------------------------------------

tensor_math_test() ->
    ?assertEqual(32.0, ar_tensor:dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])),
    ?assertEqual([14.0, 32.0], ar_tensor:matvec([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                                [1.0, 2.0, 3.0])),
    ?assertEqual(2, ar_tensor:argmax([0.1, 0.2, 0.9, 0.3])),
    P = ar_tensor:softmax([1.0, 1.0, 1.0]),
    ?assert(abs(lists:sum(P) - 1.0) < 1.0e-9).

%% ---------------------------------------------------------------------------
%% End-to-end: load model + forward + generate
%% ---------------------------------------------------------------------------

forward_shape_test() ->
    G = ar_gguf:parse(ar_synth:tiny_llama_gguf()),
    Model = ar_llama:load(G),
    ?assertEqual(6, ar_llama:vocab_size(Model)),
    Logits = ar_llama:forward(Model, [1, 2]),   %% tokens "a","b"
    ?assertEqual(6, length(Logits)),
    ?assert(lists:all(fun(X) -> is_float(X) andalso X == X end, Logits)).

generate_test() ->
    G = ar_gguf:parse(ar_synth:tiny_llama_gguf()),
    Model = ar_llama:load(G),
    Tok = ar_tokenizer:from_gguf(G),
    Opts = #{prompt => <<"ab">>, max_tokens => 5, temperature => 0.0,
             top_k => 0, seed => 42, add_bos => true},
    {Gen, Text, Stats} = ar_generate:run(Model, Tok, Opts),
    ?assert(length(Gen) >= 1),
    ?assert(length(Gen) =< 5),
    ?assert(is_binary(Text)),
    ?assertEqual(2, maps:get(prompt_tokens, Stats)).   %% [bos, "ab"] ("a"+"b" merge)
