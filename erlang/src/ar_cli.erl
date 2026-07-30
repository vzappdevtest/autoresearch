%%% Command-line interface for the pure-Erlang GGUF LLM engine.
%%% Subcommands: `info` and `generate`.
-module(ar_cli).

-export([run/1]).

run(["info", Path | Rest]) ->
    Verbose = lists:member("--verbose", Rest) orelse lists:member("-v", Rest),
    info(Path, Verbose);
run(["generate", Path | Rest]) ->
    generate(Path, parse_gen_opts(Rest, default_gen_opts()));
run(_) ->
    usage().

usage() ->
    io:format(
      "autoresearch (Erlang) - load GGUF LLMs and generate text~n~n"
      "Usage:~n"
      "  autoresearch info <model.gguf> [--verbose]~n"
      "  autoresearch generate <model.gguf> [options]~n~n"
      "Generate options:~n"
      "  -p, --prompt <text>       prompt to complete (default: \"Hello\")~n"
      "  -n, --max-tokens <n>      max tokens to generate (default: 64)~n"
      "      --temperature <f>     sampling temperature, 0 = greedy (default: 0.8)~n"
      "      --top-k <n>           top-k cutoff, 0 = off (default: 0)~n"
      "      --seed <n>            RNG seed (default: 299792458)~n"
      "      --no-bos              do not prepend the BOS token~n"),
    halt(1).

%% ---------------------------------------------------------------------------
%% info
%% ---------------------------------------------------------------------------

info(Path, Verbose) ->
    case ar_gguf:read_file(Path) of
        {error, Reason} ->
            io:format("error: cannot read ~s: ~p~n", [Path, Reason]), halt(1);
        {ok, G} ->
            Arch = to_str(ar_gguf:meta(G, <<"general.architecture">>, <<"<unknown>">>)),
            Name = to_str(ar_gguf:meta(G, <<"general.name">>, <<"<unnamed>">>)),
            Names = ar_gguf:tensor_names(G),
            io:format("File:          ~s~n", [Path]),
            io:format("Name:          ~s~n", [Name]),
            io:format("Architecture:  ~s~n", [Arch]),
            io:format("Tensors:       ~p~n", [length(Names)]),
            io:format("~nKey parameters:~n"),
            print_params(G, Arch),
            io:format("~nQuantization (tensors per dtype):~n"),
            print_quant(G, Names),
            case Verbose of
                true -> io:format("~nAll metadata: (large arrays truncated)~n"),
                        print_all_meta(G);
                false -> ok
            end
    end.

print_params(G, Arch) ->
    Params = [{"Context length", <<"context_length">>},
              {"Embedding dim", <<"embedding_length">>},
              {"Block/layer count", <<"block_count">>},
              {"FFN dim", <<"feed_forward_length">>},
              {"Attention heads", <<"attention.head_count">>},
              {"KV heads", <<"attention.head_count_kv">>},
              {"RoPE freq base", <<"rope.freq_base">>}],
    ArchB = list_to_binary(Arch),
    lists:foreach(
      fun({Label, Suffix}) ->
          Key = <<ArchB/binary, ".", Suffix/binary>>,
          case ar_gguf:meta(G, Key) of
              undefined -> ok;
              V -> io:format("  ~-18s: ~s~n", [Label, ar_gguf:value_to_string(V)])
          end
      end, Params),
    case ar_gguf:meta(G, <<"tokenizer.ggml.tokens">>) of
        undefined -> ok;
        Toks -> io:format("  ~-18s: ~p~n", ["Vocab size", length(Toks)])
    end.

print_quant(G, Names) ->
    Counts = lists:foldl(
               fun(N, Acc) ->
                   {_Dims, Type, _Off} = ar_gguf:tensor_info(G, N),
                   Dt = ar_gguf:dtype_name(Type),
                   maps:update_with(Dt, fun(C) -> C + 1 end, 1, Acc)
               end, #{}, Names),
    lists:foreach(fun({Dt, C}) -> io:format("  ~-10s: ~p~n", [Dt, C]) end,
                  lists:sort(maps:to_list(Counts))).

print_all_meta(G) ->
    Keys = lists:sort([binary_to_list(K) || K <- ar_gguf:metadata_keys(G)]),
    lists:foreach(
      fun(K) ->
          KB = list_to_binary(K),
          io:format("  ~s = ~s~n", [K, ar_gguf:value_to_string(ar_gguf:meta(G, KB))])
      end, Keys).

%% ---------------------------------------------------------------------------
%% generate
%% ---------------------------------------------------------------------------

default_gen_opts() ->
    #{prompt => "Hello", max_tokens => 64, temperature => 0.8,
      top_k => 0, seed => 299792458, add_bos => true}.

parse_gen_opts([], Opts) -> Opts;
parse_gen_opts(["-p", V | T], Opts) -> parse_gen_opts(T, Opts#{prompt => V});
parse_gen_opts(["--prompt", V | T], Opts) -> parse_gen_opts(T, Opts#{prompt => V});
parse_gen_opts(["-n", V | T], Opts) -> parse_gen_opts(T, Opts#{max_tokens => list_to_integer(V)});
parse_gen_opts(["--max-tokens", V | T], Opts) -> parse_gen_opts(T, Opts#{max_tokens => list_to_integer(V)});
parse_gen_opts(["--temperature", V | T], Opts) -> parse_gen_opts(T, Opts#{temperature => to_float(V)});
parse_gen_opts(["--top-k", V | T], Opts) -> parse_gen_opts(T, Opts#{top_k => list_to_integer(V)});
parse_gen_opts(["--seed", V | T], Opts) -> parse_gen_opts(T, Opts#{seed => list_to_integer(V)});
parse_gen_opts(["--no-bos" | T], Opts) -> parse_gen_opts(T, Opts#{add_bos => false});
parse_gen_opts([Unknown | T], Opts) ->
    io:format("warning: ignoring unknown option ~s~n", [Unknown]),
    parse_gen_opts(T, Opts).

generate(Path, Opts) ->
    case ar_gguf:read_file(Path) of
        {error, Reason} ->
            io:format("error: cannot read ~s: ~p~n", [Path, Reason]), halt(1);
        {ok, G} ->
            Arch = ar_gguf:architecture(G),
            case Arch of
                <<"llama">> -> ok;
                _ -> io:format("error: only 'llama' arch is supported for generation "
                               "(got ~s)~n", [to_str(Arch)]), halt(1)
            end,
            io:format("Loading ~s (arch: ~s) ...~n", [Path, to_str(Arch)]),
            Tok = ar_tokenizer:from_gguf(G),
            Model = ar_llama:load(G),
            io:format("Generating...~n~n"),
            {_Gen, Text, Stats} = ar_generate:run(Model, Tok, Opts),
            io:format("~s~n", [Text]),
            #{prompt_tokens := PT, generated_tokens := GT, elapsed_secs := S} = Stats,
            TokPerSec = case S > 0 of true -> GT / S; false -> 0.0 end,
            io:format("~n--- ~p prompt tokens | ~p generated in ~.2fs (~.1f tok/s)~n",
                      [PT, GT, S, TokPerSec])
    end.

%% ---------------------------------------------------------------------------
%% helpers
%% ---------------------------------------------------------------------------

to_str(B) when is_binary(B) -> binary_to_list(B);
to_str(L) when is_list(L) -> L;
to_str(undefined) -> "<unknown>".

to_float(S) ->
    case string:to_float(S) of
        {error, no_float} -> float(list_to_integer(S));
        {F, _} -> F
    end.
