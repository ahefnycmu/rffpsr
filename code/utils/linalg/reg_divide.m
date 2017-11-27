function Z = reg_divide(X, Y, lambda)
% REG_DIVIDE - Computes X/(Y + lambda * I). This implementation
% works with CPU and GPU arrays
    d = size(Y,1);
    assert(d == size(Y,2));    
    Y(1:d+1:end) = Y(1:d+1:end) + lambda;
    Z = X / Y;
    