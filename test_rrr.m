% Make data
N = 1000;
X = rand(N, 4);
Y = [X(:,1)+X(:,2), X(:,3)+0.1*X(:,4).^2, X(:,1) + 0.25*randn(N,1), randn(N,1)];

% Test unspecified case
[~, mse(1), t(1)] = rrr(X, Y);

% Test specified case
[~, mse(2), t(2)] = rrr(X, Y, 'rank', 3);

% Test correlation control case
[~, mse(3), t(3)] = rrr(X, Y, 'rank', 0.05);

% Test minimized mse case
[~, mse(4), t(4)] = rrr(X, Y, 'rank', [100, 10]);

t
mse