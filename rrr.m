function beta = rrr(X, Y, varargin)
%RRR Plot graph (nodes and edges).
%   [BETA] = RRR(X, Y) Finds the reduced rank regression using a full rank 
%   assumption. X is a n-by-r matrix, and Y is a n-by-s matrix. The rank, 
%   t, is defined as t = min(r, s). 

%   RRR(X, Y, 'PARAM1', VALUE1, 'PARAM2', VALUE2) specifies additional 
%   parameter name/value pairs chosen from the following:
%       'rank'      Specifies how to compute the apprporiate rank. Follow
%                   with an integer greater than or equal to 1 to specify 
%                   the rank of the matrix directly. Follow with a floating
%                   point number less than 1 to specify a significance
%                   value to compute the rank by linear correlation between
%                   rows in Y.  A vector of two integer [N, K] defined a
%                   K-folds cross-validation pattern to use. The
%                   appropriate rank is then estimated via minimum square
%                   error of the k-folds testing set. If not set, defaults
%                   to full rank.
%       'weights'   Define a positive-definite s-by-s weighting matrix.
%                   Default value is inv(cov(Y)).
%
%       

% Check the X and Y matrices for validity


% Define prima facie constants.
r = size(X, 2);
s = size(Y, 2);

% Handle the optimal arguments
t = min(r, s);
G = 0;
if nargin > 2
    for i=3:2:nargin
        if strcmp(varargin{i-2}, 'weights')
            G = varargin{i-1};
        end
    end

    for i=3:2:nargin
        if strcmp(varargin{i-2}, 'rank')
            if length(varargin{i-1}) == 2
                % Do cross-fold validation
            elseif varargin{i-1} >= 1
                % Just define t and run with it
                t = varargin{i-1};
            elseif varargin{i-1} < 1
                t = num_uncorr(Y, varargin{i-1});
            end
        end
    end
end

t

beta = compute_rrr(X, Y, t, G);


    function b = compute_rrr(xx, yy, tt, gg)
        % Define constants
        rr = size(xx, 2);
        nn = size(xx, 1);
        ss = size(yy, 2);
        
        full_covariance = cov([xx yy]);
        SSXX = full_covariance(1:rr, 1:rr);
        SSYX = full_covariance((rr+1):end, 1:rr);
        SSXY = full_covariance(1:rr, (rr+1):end);
        SSYY = full_covariance((rr+1):end, (rr+1):end);
        
        % Define the weighting matrix
        if length(gg) == 1
            gg = inv(SSYY);
        end

        % Define the matrix of eigen-values
        [VV, ~] = eigs(sqrtm(gg)*SSYX*inv(SSXX)*SSXY*sqrtm(gg));
        VV = fliplr(VV);
        VVt = VV(:,1:tt);

        % Define the decomposition and mean matrices
        AAt = sqrtm(inv(gg))*VVt;
        BBt = VVt'*sqrtm(gg)*SSYX*inv(SSXX);
        MMt = mean(yy)' - AAt*BBt*(mean(xx)');

        b = [MMt AAt*BBt];
    end

    function  t = num_uncorr(yy, p_crit)
        while true
            [~, p] = corr(yy);
            score = sum(min(p) < p_crit);
            if score == 0
                break;
            end
            idx = find(min(p) == min(min(p)));
            yy(:,idx(1)) = [];
        end    
        
        t = size(yy, 2);
    end
end
