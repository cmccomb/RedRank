function beta = rrr(X, Y, t)
r = size(X, 2);
n = size(X, 1);
s = size(Y, 2);

full_covariance = cov([X Y]);

SXX = full_covariance(1:r, 1:r);
SYX = full_covariance((r+1):end, 1:r);
SXY = full_covariance(1:r, (r+1):end);
SYY = full_covariance((r+1):end, (r+1):end);

G = inv(SYY);
[V, ~] = eigs(sqrtm(G)*SYX*inv(SXX)*SXY*sqrtm(G));
V = fliplr(V);
Vt = V(:,1:t);
At = sqrtm(inv(G))*Vt;
Bt = Vt'*sqrtm(G)*SYX*inv(SXX);
Mt = mean(Y)' - At*Bt*(mean(X)');

beta = [Mt At*Bt];

end
