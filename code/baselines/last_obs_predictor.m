function model = last_obs_predictor(obs, horizon, options)
    if ~isfield(options, 'range'); options.range = []; end;
        
    all_obs = flatten_features(obs, finite_future_feature_extractor(1, false), options.range);
    
    model.f0 = mean(all_obs, 2);    
    model.filter = @lobs_filter;
    model.predict = @lobs_predict;
    model.test = @lobs_test;
    model.future_win = horizon;
end

function o = lobs_predict(model, f, a)
    o = f;
end

function sf = lobs_filter(model, f, o, a)
    sf = o;
end

function o = lobs_test(model, f, a)
    o = repmat(f, 1, model.future_win); 
end

