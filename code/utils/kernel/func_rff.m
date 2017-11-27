function Z = func_rff(W, X)
%FUNC_RFF Computes random fourier features of input X using frequencies W
    Y = W * X;
    K = size(W,1);    
    Z = [cos(Y); sin(Y)] / sqrt(K);
    
    %N = size(X,2);
    %Z = zeros(2*K,N);    
    %Z(1:2:end,:) = cos(Y) / sqrt(K);
    %Z(2:2:end,:) = sin(Y) / sqrt(K);
    
    
    