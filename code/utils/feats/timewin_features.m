function F = timewin_features(X, t, win_length, delta, begin_feats)
% Returns a window of length win_length starting from t+delta
% If extra_feats flag is set, two additional dimensions are added for each
% element in the window to indicate "before beginnig" and "after end".
    d = size(X, 1);
    T = size(X, 2);
    
    t = t+delta;
    
    prefix = min(max(0, 1-t), win_length);
    suffix = min(max(0, t+win_length-1-T), win_length);
    data = max(min(min(T-t+1, win_length - (1-t)), win_length), 0); 
    
    t = max(t,1);
        
    F = [zeros(d, prefix) X(:, t:t+data-1) zeros(d, suffix)];
    
    if begin_feats
        F = [F; ones(1, prefix) zeros(1, data + suffix)]; %; zeros(1, prefix + data) ones(1, suffix)];
        d = d+1;
    end
    
    F = reshape(F, d * win_length, 1);
end
