function [Y, mean_X, std_X] = normalize_sequences(X)
    N = 0;
    d = size(X{1},1);
    sum_X = zeros(d,1);
    sum_X2 = zeros(d,1);
    
    for i = 1:length(X)
        N = N + length(X{i});
        sum_X = sum_X + sum(X{i},2);
        sum_X2 = sum_X2 + sum(X{i}.*X{i},2);
    end
    
    mean_X = sum_X / N;
    std_X = sqrt(sum_X2/N - mean_X .* mean_X);
    
    Y = cell(1,length(X));
    
    for i = 1:length(X)
        Y{i} = bsxfun(@minus, X{i}, mean_X);
        Y{i} = bsxfun(@times, X{i}, 1./ std_X);
    end
end
