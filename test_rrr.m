% Make data
N = 1000;
X = rand(N, 4);
Y = [X(:,1)+X(:,2), X(:,3)+0.1*X(:,4).^2, X(:,1) + 0.25*randn(N,1), randn(N,1)];

% Test unspecified case
[~, mse, t] = rrr(X, Y);
fprintf('Leave t full rank:\n\tMSE = %.3f\n\tt = %d\n', mse, t);

% Test specified case
[~, mse, t] = rrr(X, Y, 'rank', 2);
fprintf('Specify the value of t:\n\tMSE = %.3f\n\tt = %d\n', mse, t);

% Test correlation control case
[~, mse, t] = rrr(X, Y, 'rank', 0.05);
fprintf('Select t with correlation analysis:\n\tMSE = %.3f\n\tt = %d\n', mse, t);

% Test minimized mse case
[~, mse, t] = rrr(X, Y, 'rank', [100, 5]);
fprintf('Select t with MSE minimization:\n\tMSE = %.3f\n\tt = %d\n', mse, t);

% Test unspecified case
w = eye(4); w(1,1) = 0.01; w(2,2) = 100;
[~, mse, t] = rrr(X, Y, 'weighting', w);
fprintf('Leave t full rank with skew weighting:\n\tMSE = %.3f\n\tt = %d\n', mse, t);

% Test specified case
[~, mse, t] = rrr(X, Y, 'rank', 2, 'weighting', w);
fprintf('Specify the value of t with skew weighting:\n\tMSE = %.3f\n\tt = %d\n', mse, t);

% Test correlation control case
[~, mse, t] = rrr(X, Y, 'rank', 0.05, 'weighting', w);
fprintf('Select t with correlation analysis with skew weighting:\n\tMSE = %.3f\n\tt = %d\n', mse, t);

% Test minimized mse case
[~, mse, t] = rrr(X, Y, 'rank', [100, 5], 'weighting', w);
fprintf('Select t with MSE minimization with skew weighting:\n\tMSE = %.3f\n\tt = %d\n', mse, t);