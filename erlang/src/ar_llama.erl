%%% Llama-architecture forward pass in pure Erlang.
%%%
%%% Implements the standard decoder-only transformer used by GGUF `llama` models
%%% (Llama 1/2/3, Mistral, TinyLlama, ...): token embedding, per-layer RMSNorm,
%%% grouped-query attention with RoPE and a causal mask, SwiGLU feed-forward,
%%% a final norm and the output projection.
%%%
%%% For simplicity the whole token sequence is recomputed on each step (no KV
%%% cache), which is fine for the small models this pure-Erlang engine targets.
-module(ar_llama).

-export([load/1, forward/2, config/1, vocab_size/1]).

-record(model, {cfg :: map(), w :: map()}).

%% ---------------------------------------------------------------------------
%% Loading
%% ---------------------------------------------------------------------------

%% Build an in-memory model (config + dequantized weights) from a parsed GGUF.
-spec load(ar_gguf:gguf()) -> #model{}.
load(G) ->
    <<"llama">> = ar_gguf:architecture(G),   %% assert llama arch
    NHead = ar_gguf:meta(G, <<"llama.attention.head_count">>),
    NKV = ar_gguf:meta(G, <<"llama.attention.head_count_kv">>),
    NLayer = ar_gguf:meta(G, <<"llama.block_count">>),
    Embd = ar_gguf:meta(G, <<"llama.embedding_length">>),
    HeadDim = Embd div NHead,
    RopeDim = ar_gguf:meta(G, <<"llama.rope.dimension_count">>, HeadDim),
    Eps = ar_gguf:meta(G, <<"llama.attention.layer_norm_rms_epsilon">>, 1.0e-5),
    FreqBase = ar_gguf:meta(G, <<"llama.rope.freq_base">>, 10000.0),
    Cfg = #{n_head => NHead, n_kv_head => NKV, n_layer => NLayer,
            embd => Embd, head_dim => HeadDim, rope_dim => RopeDim,
            eps => Eps + 0.0, freq_base => FreqBase + 0.0},

    TokEmbd = ar_gguf:tensor_rows(G, <<"token_embd.weight">>),
    OutNorm = ar_gguf:tensor_f32(G, <<"output_norm.weight">>),
    Output = case lists:member(<<"output.weight">>, ar_gguf:tensor_names(G)) of
                 true -> ar_gguf:tensor_rows(G, <<"output.weight">>);
                 false -> TokEmbd   %% tied embeddings
             end,
    Layers = [load_layer(G, I) || I <- lists:seq(0, NLayer - 1)],
    W = #{tok_embd => TokEmbd, out_norm => OutNorm, output => Output,
          layers => Layers},
    #model{cfg = Cfg, w = W}.

load_layer(G, I) ->
    P = <<"blk.", (integer_to_binary(I))/binary, ".">>,
    #{attn_norm => ar_gguf:tensor_f32(G, <<P/binary, "attn_norm.weight">>),
      ffn_norm => ar_gguf:tensor_f32(G, <<P/binary, "ffn_norm.weight">>),
      wq => ar_gguf:tensor_rows(G, <<P/binary, "attn_q.weight">>),
      wk => ar_gguf:tensor_rows(G, <<P/binary, "attn_k.weight">>),
      wv => ar_gguf:tensor_rows(G, <<P/binary, "attn_v.weight">>),
      wo => ar_gguf:tensor_rows(G, <<P/binary, "attn_output.weight">>),
      w_gate => ar_gguf:tensor_rows(G, <<P/binary, "ffn_gate.weight">>),
      w_up => ar_gguf:tensor_rows(G, <<P/binary, "ffn_up.weight">>),
      w_down => ar_gguf:tensor_rows(G, <<P/binary, "ffn_down.weight">>)}.

config(#model{cfg = Cfg}) -> Cfg.
vocab_size(#model{w = #{tok_embd := T}}) -> length(T).

%% ---------------------------------------------------------------------------
%% Forward pass
%% ---------------------------------------------------------------------------

%% Given a list of token ids, return the logits (list of floats, length vocab)
%% for the final position.
-spec forward(#model{}, [non_neg_integer()]) -> [float()].
forward(#model{cfg = Cfg, w = W}, Tokens) ->
    #{tok_embd := TokEmbd, out_norm := OutNorm, output := Output,
      layers := Layers} = W,
    Xs0 = [lists:nth(Id + 1, TokEmbd) || Id <- Tokens],
    Xs = lists:foldl(fun(L, Acc) -> layer(L, Acc, Cfg) end, Xs0, Layers),
    #{eps := Eps} = Cfg,
    Last = lists:last(Xs),
    Xf = ar_tensor:rmsnorm(Last, OutNorm, Eps),
    ar_tensor:matvec(Output, Xf).

layer(L, Xs, Cfg) ->
    #{eps := Eps, n_head := NHead, n_kv_head := NKV, head_dim := HeadDim,
      rope_dim := RopeDim, freq_base := FreqBase} = Cfg,
    #{attn_norm := AN, ffn_norm := FN, wq := Wq, wk := Wk, wv := Wv, wo := Wo,
      w_gate := Wg, w_up := Wu, w_down := Wd} = L,

    %% --- attention ---
    Normed = [ar_tensor:rmsnorm(X, AN, Eps) || X <- Xs],
    Positions = lists:seq(0, length(Xs) - 1),
    %% Per position: project, RoPE, then split into heads.
    QHeads = [split_heads(rope(ar_tensor:matvec(Wq, H), Pos, HeadDim, RopeDim,
                                FreqBase, NHead), HeadDim)
              || {Pos, H} <- lists:zip(Positions, Normed)],
    KHeads = [split_heads(rope(ar_tensor:matvec(Wk, H), Pos, HeadDim, RopeDim,
                                FreqBase, NKV), HeadDim)
              || {Pos, H} <- lists:zip(Positions, Normed)],
    VHeads = [split_heads(ar_tensor:matvec(Wv, H), HeadDim) || H <- Normed],

    Group = NHead div NKV,
    Attn = [attention(I, QHeads, KHeads, VHeads, NHead, Group, HeadDim)
            || I <- Positions],
    Os = [ar_tensor:matvec(Wo, A) || A <- Attn],
    Xs1 = lists:zipwith(fun ar_tensor:add/2, Xs, Os),

    %% --- feed-forward (SwiGLU) ---
    Ffn = [begin
               H2 = ar_tensor:rmsnorm(X, FN, Eps),
               Gate = ar_tensor:silu(ar_tensor:matvec(Wg, H2)),
               Up = ar_tensor:matvec(Wu, H2),
               ar_tensor:matvec(Wd, ar_tensor:mul(Gate, Up))
           end || X <- Xs1],
    lists:zipwith(fun ar_tensor:add/2, Xs1, Ffn).

%% Attention for query position I over keys/values 0..I (causal).
attention(I, QHeads, KHeads, VHeads, NHead, Group, HeadDim) ->
    QI = lists:nth(I + 1, QHeads),           %% NHead head-vectors
    KUpto = lists:sublist(KHeads, I + 1),    %% positions 0..I
    VUpto = lists:sublist(VHeads, I + 1),
    Scale = 1.0 / math:sqrt(HeadDim),
    Heads = [begin
                 KvH = H div Group,
                 Qh = lists:nth(H + 1, QI),
                 KhsJ = [lists:nth(KvH + 1, Kj) || Kj <- KUpto],
                 VhsJ = [lists:nth(KvH + 1, Vj) || Vj <- VUpto],
                 Scores = [ar_tensor:dot(Qh, Kh) * Scale || Kh <- KhsJ],
                 Weights = ar_tensor:softmax(Scores),
                 weighted_sum(Weights, VhsJ, HeadDim)
             end || H <- lists:seq(0, NHead - 1)],
    lists:append(Heads).

%% Split a length-(NumHeads*HeadDim) vector into a list of head-vectors.
split_heads([], _HeadDim) -> [];
split_heads(Vec, HeadDim) ->
    {Head, Rest} = lists:split(HeadDim, Vec),
    [Head | split_heads(Rest, HeadDim)].

%% Sum_j W[j] * V[j], where each V[j] is a length-HeadDim vector.
weighted_sum(Weights, Vecs, HeadDim) ->
    Zero = lists:duplicate(HeadDim, 0.0),
    lists:foldl(fun({Wt, V}, Acc) -> ar_tensor:add(Acc, ar_tensor:scale(V, Wt)) end,
                Zero, lists:zip(Weights, Vecs)).

%% Apply RoPE (NeoX half-split convention) to each head of a projected vector.
rope(Vec, Pos, HeadDim, RopeDim, FreqBase, NumHeads) ->
    Heads = split_heads(Vec, HeadDim),
    NumHeads = length(Heads),   %% assert
    lists:append([rope_head(Hd, Pos, RopeDim, FreqBase) || Hd <- Heads]).

rope_head(Head, Pos, RopeDim, FreqBase) ->
    {RopePart, Tail} = lists:split(RopeDim, Head),
    Half = RopeDim div 2,
    {A, B} = lists:split(Half, RopePart),
    Idx = lists:seq(0, Half - 1),
    Rotated = [begin
                   Theta = Pos / math:pow(FreqBase, (2 * I) / RopeDim),
                   Cos = math:cos(Theta),
                   Sin = math:sin(Theta),
                   X0 = lists:nth(I + 1, A),
                   X1 = lists:nth(I + 1, B),
                   {X0 * Cos - X1 * Sin, X1 * Cos + X0 * Sin}
               end || I <- Idx],
    NewA = [Y0 || {Y0, _} <- Rotated],
    NewB = [Y1 || {_, Y1} <- Rotated],
    NewA ++ NewB ++ Tail.
