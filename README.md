# RedRank: reduced-rank multivariate regression for MATLAB

`rrr(X, Y, 'rank', t)` estimates a multivariate linear regression with
coefficient rank `t`. `X` and `Y` have observations in rows. It returns a
response-by-predictor coefficient matrix with an intercept in the first
column, plus mean squared error and the selected rank.

```matlab
[beta, mse, rank] = rrr(X, Y, 'rank', 2);
prediction = [ones(size(X, 1), 1) X] * beta.';
```

The core fixed-rank path works in MATLAB without additional toolboxes.
By default, the implementation weights responses with the inverse sample
covariance of `Y`; supply `'weighting', eye(size(Y,2))` for unweighted
regression. The covariance matrices must be nonsingular.

The `'rank'` modes that choose a rank through correlation or
cross-validation call `corr` or `randsample` and require Statistics and
Machine Learning Toolbox. The original `test_rrr.m` is a demonstration
of those optional modes.

Run `test_rrr_core` to check the fixed-rank path. In particular, it
verifies the File Exchange reviewer's reported failure above rank six
and the selection of the largest eigenvalue components.

This repository restores the source published in the [File Exchange
1.1.0 package](https://www.mathworks.com/matlabcentral/fileexchange/53024-reduced-rank-regression).
The historical package used the MIT license; its original notice is
retained in `LEGACY_LICENSE.md`. New work in this repository is BSD 3-Clause.

The implementation follows A. J. Izenman, “Reduced-rank regression for
the multivariate linear model,” *Journal of Multivariate Analysis*,
1975. The original File Exchange description misspelled the surname.
