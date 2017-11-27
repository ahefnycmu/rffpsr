function W = ridge_regression(X, Y, lambda, gpu, weights)

if nargin < 4; gpu = 0; end;
if nargin < 5; weights = []; end;

if gpu
    X = gpuArray(X);
    Y = gpuArray(Y);
end

if isempty(weights)
    Xw = X;
else
    Xw = bsxfun(@times, X, weights);
end

d = size(X,1);
C = Xw*X'; C(1:d+1:end) = C(1:d+1:end) + lambda;
W = (Y * Xw') / C;

if gpu
    W = gather(W);
end


