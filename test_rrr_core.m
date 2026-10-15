% Test the 2017 File Exchange review's eigenvector-ordering and rank limits.
rng(1);
raw = randn(100, 8);
raw = raw - mean(raw, 1);
[q, ~] = qr(raw, 0);
X = q * sqrt(99);  % cov(X) = I
Y = X * diag(8:-1:1);

[one, mseOne, rankOne] = rrr(X, Y, 'rank', 1, 'weighting', eye(8));
assert(rankOne == 1);
expectedOne = diag([8 zeros(1, 7)]);
assert(norm(one(:, 2:end) - expectedOne, 'fro') < 1e-8);

[seven, mseSeven, rankSeven] = rrr(X, Y, 'rank', 7, 'weighting', eye(8));
assert(rankSeven == 7);
expectedSeven = diag([8:-1:2 0]);
assert(norm(seven(:, 2:end) - expectedSeven, 'fro') < 1e-8);
assert(mseSeven < mseOne);

[full, mseFull, rankFull] = rrr(X, Y, 'rank', 8, 'weighting', eye(8));
assert(rankFull == 8);
assert(norm(full(:, 2:end) - diag(8:-1:1), 'fro') < 1e-8);
assert(mseFull < 1e-20);
