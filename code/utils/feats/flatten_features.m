function [ Xf, series_index, time_index ] = flatten_features( X, feature_extractor, range )
    if nargin < 3; range = []; end;
    begin_cut = 0;
    end_cut = 0;
    
    if length(range) == 2 && max(range) <= 0
        begin_cut = -range(1);
        end_cut = -range(2);
        range = [];
    end
    
    
    d = size(feature_extractor(X{1}, 1), 1);    
    N = 0;
    
    for i=1:length(X)        
        if isempty(range); T = size(X{i}, 2); 
        else T = length(range); end;
        N = N + T - begin_cut - end_cut;
    end
    
    Xf = zeros(d, N);
    series_index = zeros(1, N);
    time_index = zeros(1, N);
        
    n = 1;
    for i=1:length(X)
        %fprintf('Flattening sequence %d\n', i);
        T = size(X{i}, 2);
        if isempty(range); range_i = 1+begin_cut:T-end_cut; 
        else range_i = range; end;
        
        for t=range_i
            Xf(:,n) = feature_extractor(X{i}, t);            
            series_index(n) = i;
            time_index(n) = t;
            n = n+1;
        end
    end
end

