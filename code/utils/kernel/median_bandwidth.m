function [ h ] = median_bandwidth( X, max_points )
[~,n] = size(X);
if nargin < 2 || isempty(max_points); max_points = n; end;

if n > max_points
    idx = randsample(n, max_points, false);
    X = X(:,idx);
end

D = fast_pdist2(X);
h = sqrt(median(D));    
end

function D = fast_pdist2(X)
    n = size(X,2);
    XX = repmat(sum(X.*X, 1), n, 1);
    D = XX + XX' - 2*(X'*X);
    
    i = 0:n*n-1;
    j = floor(i/n) < mod(i,n);
    D = D(j);
end


