% Make data
X = rand(100, 4);
Y = [1:100; (1:100).^2; rand(1,100); randn(1,100)]';

% Test unspecified case
rrr(X, Y)

% Test specified case
rrr(X, Y, 'rank', 2);

% Test linear case
rrr(X, Y, 'rank', 0.05);