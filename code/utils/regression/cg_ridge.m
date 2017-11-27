function W = cg_ridge(X, Y, lambda, options)
%CG_RIDGE - Ridge regression using conjugate gradient
%Parameters: 
% X,Y - Training inputs and outputs (each column is a data point)
% lambda - Ridge regularization parameter. The function solves the problem:
%  min_W || WX - Y ||^2 + (lambda/2) * ||W||^2_F
% options - Structure of additional options.
    if nargin < 4; options = struct; end;
    if ~isfield(options, 'maxit'); options.maxit = 1000; end;   % Maximum iterations
    if ~isfield(options, 'eps'); options.eps = 1e-5; end;       % Relative residual tolerance
    if ~isfield(options, 'gpu'); options.gpu = 0; end;          % Whether to use GPU for computation    
    if ~isfield(options, 'W0'); options.W0 = []; end;
    
    [d_in,n] = size(X);            
    d_out = size(Y,1);    
    
    if options.gpu 
        X = gpuArray(X);
        Y = gpuArray(Y);        
    end
                
    if isempty(options.W0) && d_in > n && d_out * d_in > n * d_in                
        % Use the dual (gram) formulation        
        W0 = zeros(n * d_in, 1);
        XT = reshape(X', [], 1);
        G = X' * X;
        A = @(w) reshape(G * reshape(w, n, d_in), [], 1) + lambda * w;        
        W = pcg(A, XT, options.eps, options.maxit, [], [], W0);
        W = reshape(W, n, d_in);
        W = Y * W;
    else
        if isempty(options.W0); options.W0 = zeros(d_out, d_in); end;    
        W0 = reshape(options.W0, [], 1);
        if options.gpu; W0 = gpuArray(W0); end;
        
        XY = reshape(X * Y', [], 1);        
        A = @(w) reshape(X * (X' * reshape(w, d_in, [])), [], 1) + lambda * w;        
        W = pcg(A, XY, options.eps, options.maxit, [], [], W0);
        W = reshape(W, d_in, []);
        W = W';
    end
    
    if options.gpu
        W = gather(W);
    end
    
    
    
        
    
       
        
        