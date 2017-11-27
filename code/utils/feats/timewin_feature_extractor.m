function extractor = timewin_feature_extractor(win_length, delta, extra_feats)
    if nargin < 3 || isempty(extra_feats); extra_feats = 0; end;    
    extractor = @(X, t) timewin_features(X, t, win_length, delta, extra_feats);
end

