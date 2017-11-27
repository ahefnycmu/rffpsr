function [U, S, UX] = rand_svd_f(f, n, k, it, slack, blk)
%RAND_SVD_F Computes randomized left singular vectors of a dxn matrix X represented by a column sampling function f. 
%The method uses randomized SVD algorithm by Halko, Martinson and Tropp.
%Parameters:
% f - Column sampling function: f(s,e) should return columns s through e of the input matrix.
% n - Number of columns of the input matrix.
% k - Maximum number of songular vectors.
% it - (optional, default=2) The exponent q to premultiply (XX')^q by X to suppress small eigen values.
% slack - (optional, default=0) Extra dimensions to use in intermediate steps.
% blk (optional, default=1000) - Number of input columns that can be stored in memory.
% Outputs:
% U - Singular vectors.
% S - Singular values.
% UX - Projected input matrix U' * X

if nargin < 4 || isempty(it); it = 2; end
if nargin < 5 || isempty(slack); slack = 0; end   
if nargin < 6 || isempty(blk); blk = 1000; end   

x = f(1,1);
d = size(x,1);
gpu = isa(x, 'gpuArray');

if d <= k
    % No need for SVD
    U = speye(d);
    S = ones(1,d); % Dummy values
    if nargout == 3; UX = f(1,n); end;
    return;
end

p = k + slack;
num_blocks = ceil(n/blk);

K = zeros(d,p);
Pb = randn(n,p);
if gpu
    K = def_gpuArray(K);
    Pb = def_gpuArray(Pb);
end

for b = 1:num_blocks
    blk_start = (b-1)*blk+1;
    blk_end = min(n, blk_start+blk-1);
    Xb = f(blk_start, blk_end);
    %Pb = randn(size(Xb,2),p);
    K = K + Xb * Pb(blk_start:blk_end,:);        
end

for i = 1:it
    KK = zeros(d,p);
    if gpu; KK = def_gpuArray(KK); end;
    
    for b = 1:num_blocks
        blk_start = (b-1)*blk+1;
        blk_end = min(n, blk_start+blk-1);
        
        Xb = f(blk_start, blk_end);
        KK = KK + Xb * (Xb' * K);
    end
    
    K = KK / max(abs(KK(:)));
end

Q = orth(K);
p = size(Q,2);

if nargout == 3; 
    qx = zeros(p,n); 
    if gpu; qx = def_gpuArray(qx); end;
end;

M = zeros(p,p);
if gpu; M = def_gpuArray(M); end;

for b = 1:num_blocks
    blk_start = (b-1)*blk+1;
    blk_end = min(n, blk_start+blk-1);
    qxb = Q' * f(blk_start, blk_end);
    M = M + qxb * qxb';
    
    if nargout == 3; qx(:,blk_start:blk_end) = qxb; end;
end

[Um,S,~] = svd(M);
if k < size(Um,2)
    Um = Um(:,1:k);
    S = S(1:k,1:k);
end

U = Q * Um;
S = sqrt(diag(S));

if nargout == 3; UX = Um' * qx; end;

end




