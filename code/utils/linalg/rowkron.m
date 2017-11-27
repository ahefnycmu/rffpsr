function [ z ] = rowkron(x, y)
%ROWKRON Kronecker product of two row vectors. 
%For some reason, this is much faster than MATLAB's 'kron'
    z = reshape(y' * x, 1, []);
end

