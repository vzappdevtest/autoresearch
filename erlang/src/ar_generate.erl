%%% Autoregressive generation loop for the pure-Erlang llama engine.
-module(ar_generate).

-export([run/3, sample/4]).

%% Options map:
%%   prompt      => binary()|string()
%%   max_tokens  => integer()
%%   temperature => float()   (0.0 => greedy/argmax)
%%   top_k       => integer() (0 => disabled)
%%   seed        => integer()
%%   add_bos     => boolean()
%%
%% Returns {GeneratedIds, DecodedText, Stats} where Stats is a map with token
%% counts and elapsed seconds.
-spec run(term(), ar_tokenizer:tokenizer(), map()) ->
          {[integer()], binary(), map()}.
run(Model, Tok, Opts) ->
    Prompt = maps:get(prompt, Opts),
    Max = maps:get(max_tokens, Opts, 64),
    Temp = maps:get(temperature, Opts, 0.8) + 0.0,
    TopK = maps:get(top_k, Opts, 0),
    Seed = maps:get(seed, Opts, 299792458),
    AddBos = maps:get(add_bos, Opts, true),

    Context0 = ar_tokenizer:encode(Tok, Prompt, AddBos),
    Eos = ar_tokenizer:eos_id(Tok),
    RS0 = rand:seed_s(exsss, {Seed, Seed bxor 16#9E3779B9, (Seed bsl 1) + 1}),

    T0 = erlang:monotonic_time(millisecond),
    {Gen, _RS} = loop(Model, Context0, [], Max, Temp, TopK, Eos, RS0),
    T1 = erlang:monotonic_time(millisecond),

    Text = ar_tokenizer:decode(Tok, Gen, true),
    Stats = #{prompt_tokens => length(Context0),
              generated_tokens => length(Gen),
              elapsed_secs => (T1 - T0) / 1000.0},
    {Gen, Text, Stats}.

loop(_Model, _Ctx, Gen, 0, _Temp, _TopK, _Eos, RS) ->
    {lists:reverse(Gen), RS};
loop(Model, Ctx, Gen, N, Temp, TopK, Eos, RS) ->
    Logits = ar_llama:forward(Model, Ctx),
    {Next, RS1} = sample(Logits, Temp, TopK, RS),
    case Next =:= Eos of
        true -> {lists:reverse(Gen), RS1};
        false -> loop(Model, Ctx ++ [Next], [Next | Gen], N - 1, Temp, TopK, Eos, RS1)
    end.

%% Sample a token id from logits. Greedy when Temp =< 0.
-spec sample([float()], float(), integer(), term()) -> {integer(), term()}.
sample(Logits, Temp, _TopK, RS) when Temp =< 0.0 ->
    {ar_tensor:argmax(Logits), RS};
sample(Logits, Temp, TopK, RS) ->
    Scaled = [L / Temp || L <- Logits],
    Probs = ar_tensor:softmax(Scaled),
    Indexed = lists:zip(lists:seq(0, length(Probs) - 1), Probs),
    Filtered = apply_top_k(Indexed, TopK),
    Total = lists:sum([P || {_, P} <- Filtered]),
    {R, RS1} = rand:uniform_s(RS),
    Target = R * Total,
    {pick(Filtered, Target), RS1}.

apply_top_k(Indexed, K) when is_integer(K), K > 0, K < length(Indexed) ->
    Sorted = lists:sort(fun({_, A}, {_, B}) -> A >= B end, Indexed),
    lists:sublist(Sorted, K);
apply_top_k(Indexed, _K) ->
    Indexed.

pick([{Id, _P}], _Target) -> Id;
pick([{Id, P} | Rest], Target) ->
    case Target =< P of
        true -> Id;
        false -> pick(Rest, Target - P)
    end.
