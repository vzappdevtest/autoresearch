%%% Minimal tensor / vector math for pure-Erlang LLM inference.
%%% Vectors are lists of floats; matrices are lists of row-vectors.
%%% Not fast — intended for small models and clarity, not production scale.
-module(ar_tensor).

-export([dot/2, matvec/2, add/2, mul/2, scale/2, rmsnorm/3, softmax/1,
         silu/1, argmax/1]).

%% Dot product of two equal-length vectors.
-spec dot([float()], [float()]) -> float().
dot(A, B) -> dot(A, B, 0.0).
dot([], [], Acc) -> Acc;
dot([A | As], [B | Bs], Acc) -> dot(As, Bs, Acc + A * B).

%% Matrix (list of rows) times a vector -> vector (one dot per row).
-spec matvec([[float()]], [float()]) -> [float()].
matvec(Rows, X) -> [dot(R, X) || R <- Rows].

%% Elementwise add / multiply.
add(A, B) -> lists:zipwith(fun(X, Y) -> X + Y end, A, B).
mul(A, B) -> lists:zipwith(fun(X, Y) -> X * Y end, A, B).

%% Scale a vector by a scalar.
scale(A, S) -> [X * S || X <- A].

%% RMSNorm: x / sqrt(mean(x^2) + eps) elementwise, then * weight.
-spec rmsnorm([float()], [float()], float()) -> [float()].
rmsnorm(X, Weight, Eps) ->
    N = length(X),
    SumSq = lists:foldl(fun(V, A) -> A + V * V end, 0.0, X),
    Scale = 1.0 / math:sqrt(SumSq / N + Eps),
    lists:zipwith(fun(V, W) -> V * Scale * W end, X, Weight).

%% Numerically-stable softmax over a list.
-spec softmax([float()]) -> [float()].
softmax(Xs) ->
    Max = lists:max(Xs),
    Exps = [math:exp(X - Max) || X <- Xs],
    Sum = lists:sum(Exps),
    [E / Sum || E <- Exps].

%% SiLU / swish activation: x * sigmoid(x).
-spec silu([float()]) -> [float()].
silu(Xs) -> [X / (1.0 + math:exp(-X)) || X <- Xs].

%% Index of the maximum element (0-based).
-spec argmax([float()]) -> non_neg_integer().
argmax([H | T]) -> argmax(T, 1, 0, H).
argmax([], _I, BestI, _Best) -> BestI;
argmax([H | T], I, _BestI, Best) when H > Best -> argmax(T, I + 1, I, H);
argmax([_H | T], I, BestI, Best) -> argmax(T, I + 1, BestI, Best).
