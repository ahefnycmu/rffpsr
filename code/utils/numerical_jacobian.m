function J = numerical_jacobian(d, f, x, h)        
    J = zeros(d, length(x));
    for i = 1:length(x)
        old_x = x(i);
        x(i) = x(i) + h;
        fp = f(x);
        x(i) = x(i) - 2*h;
        fm = f(x);
        x(i) = old_x;
        
        J(:,i) = (fp-fm)/(2*h);
    end
end