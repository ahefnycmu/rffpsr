function validate_jacobian(d, f, g, h, num_trials, x)
% VALIDATE_JACOBIAN: Function to check the Jacobian of a vector-valued
% function. The function tests the finite difference and the first order
% approximation along randomly chosen directions.
% Parameters:
%   d - Input dimension
%   f - Handle to function
%   g - Handle to Jacobian function. For a given vector X, the function
%   returns the Jacobian matrix (Note: for scalar functions, this is the 
%   transpose of the gradient).
%   h - finite difference
%   num_trials - Number of tests.
%   x - A point sampling function or a fixed point to test at.
%   If not provided, a random Gaussian point is chosen for each test.
    if nargin == 0
        unit_test()
        return
    end
    
    if nargin < 5; num_trials = 10; end    
    if nargin < 6; x = @() randn(d,1); end
    
    if isa(x, 'function_handle')
        gen_x = x;
    else
        gen_x = @() x;
    end
        
    disp('Validating Jacobian');
    max_rel_err = -Inf;
    max_abs_err = -Inf;
    
    for t = 1:num_trials
        x = gen_x();        
        gx = g(x);

        % Pick a random direction
        delta = randn(d,1);
        delta = delta / norm(delta);
        
        % Report error in gradient approximation in the direction of delta
        df = (f(x+delta*h) - f(x-delta*h))/(2*h);
        dfh = gx*delta;
        assert(size(df,1) == size(dfh,1));
        assert(size(df,2) == size(dfh,2));
        abs_err = norm(df - dfh, 'fro');
        rel_err = abs_err / norm(df, 'fro');
        fprintf('Test:%d abs_error=%e rel_error=%e\n', t, abs_err, rel_err);        
        
        max_abs_err = max(max_abs_err, abs_err);
        max_rel_err = max(max_rel_err, rel_err);
    end
        
    fprintf('Max error: abs=%e rel=%e\n', max_abs_err, max_rel_err);
end

function unit_test()    
    f = @(x) [x(1)^3+2*x(1)*x(2)+x(2)^2; x(1)];
    g = @(x) [3*x(1)^2 + 2*x(2) 2*x(1)+2*x(2); 1 0];
    h = 1e-10;
    
    % Example with a given point.
    validate_jacobian(2, f, g, h, 10, [1; 0]);
    
    % Example with random points.
    validate_jacobian(2, f, g, h, 10);
end