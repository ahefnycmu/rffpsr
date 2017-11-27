function psr = train_rff_ar(obs, act, future_win, past_win, ...
                                   prediction_tasks, options)
%TRAIN_RFF_AR - Trains an auto regressive model using RFF features.

    if nargin < 5 || isempty(prediction_tasks); prediction_tasks = {}; end;

    if nargin < 6 || isempty(options); options = struct; end;
    if ~isfield(options, 'reg_gpu'); options.reg_gpu = 0; end; % Use GPU for regression
    if ~isfield(options, 'reg_maxit'); options.reg_maxit = 1000; end; % Max. iterations for regression optimization algorithms
    if ~isfield(options, 'D'); options.D = 5000; end;
    if ~isfield(options, 'p'); options.p = 200; end;
    if ~isfield(options, 'lambda'); options.lambda = 1e-3; end;
    if ~isfield(options, 'range'); options.range = [-past_win -future_win]; end; 
    if ~isfield(options, 'discount'); options.discount = 0.8; end;
    
    range = options.range;            
    lambda = options.lambda;
    D = options.D;
    K = options.p;
        
    k = future_win;
    d_o = size(obs{1}, 1);
    d_a = size(act{1}, 1);
    
    all_test_o = flatten_features(obs, finite_future_feature_extractor(k, false), range);
    all_test_a = flatten_features(act, finite_future_feature_extractor(k, false), range);            
    d_to = size(all_test_o,1);
    d_ta = size(all_test_a,1);
    
    all_past_obs = flatten_features(obs, finite_past_feature_extractor(past_win, false), range);
    all_past_act = flatten_features(act, finite_past_feature_extractor(past_win, false), range);    
    all_past_obs = discount(all_past_obs, d_o, options.discount);
    all_past_act = discount(all_past_act, d_a, options.discount);
        
    all_past = [all_past_obs; all_past_act];                       
    d_h = size(all_past, 1);    
        
    all_obs = all_test_o(1:d_o,:);
    all_act = all_test_a(1:d_a,:);            
    
    all_1sp_in = [all_past; all_act]; % input for 1-step prediction
    all_tp_in = [all_past; all_test_a]; % input for test prediction
        
    N = size(all_test_o, 2);
    
    %% Transform training data using random Fourier features       
    
    % Compute kernel bandwidths using median trick
    disp('Estimating Kernel Bandwidths');
    tic;
    s_1sp = median_bandwidth(all_1sp_in, 5000);
    s_tp = median_bandwidth(all_tp_in, 5000);
    toc;
    
    % Sample from kernel spectra
    tic;
    disp('Sampling Frequencies');
    V_1sp = randn(D, d_h+d_a) / s_1sp;      
    V_tp = randn(D, d_h+d_ta) / s_tp;      
    toc;
    
    % Create feature extractors
    feat_1sp = @(s,e) func_rff(V_1sp, all_1sp_in(:,s:e));
    feat_tp = @(s,e) func_rff(V_tp, all_tp_in(:,s:e));
    
    disp('Computing low dim representation of regression inputs');
    tic;        
    [U_1sp,~, all_1sp_feat] = rand_svd_f(feat_1sp,N,K,[],50);        
    K_1sp = size(all_1sp_feat,1);                  
    [U_tp,~, all_tp_feat] = rand_svd_f(feat_tp,N,K,[],50);        
    K_tp = size(all_tp_feat,1);                  
    toc;            
                            
    %% Regression                       
    % Regression for prediction:    
    disp('Regression: 1-Step Prediction');
    tic;
    s2p_in = all_1sp_feat;
    s2p_out = all_obs;
    reg_options.gpu = options.reg_gpu;
    reg_options.maxit = options.reg_maxit;
    W_2p = ridge_regression(s2p_in, s2p_out, lambda, reg_options.gpu);    
    toc;

    disp('Regression: Test Prediction');
    tic;
    s2t_in = all_tp_feat;
    s2t_out = all_test_o;
    W_2t = ridge_regression(s2t_in, s2t_out, lambda, reg_options.gpu);    
    toc;

    disp('S2 Regression: Additional Prediction Tasks');
    tic;
    num_tasks = length(prediction_tasks);
    W_2tsk = cell(1,num_tasks);

    for task = 1:num_tasks
        s2tsk_in = s2p_in;
        s2tsk_out = flatten_features(prediction_tasks{task}, @(x,t) ...
                                     x(t), range);
        W_2tsk{task} = cg_ridge(s2tsk_in, s2tsk_out, lambda, reg_options);    
    end
    
    toc;
        
    %% Build Model
    % Private data members    
    psr.Wp_ = W_2p;
    psr.Wt_ = W_2t;
    psr.Wtsk_ = W_2tsk;    
    psr.V_1sp_ = V_1sp;
    psr.V_tp_ = V_tp;
    psr.U_1sp_ = U_1sp;    
    psr.U_tp_ = U_tp;    
    psr.d_o_ = d_o;
    psr.d_a_ = d_a;
    psr.D_ = D;
    psr.discount_ = options.discount;
    
    % Public data memebers
    
    task_dim = cell(1,num_tasks);

    for task = 1:num_tasks           
        task_dim{task} = size(prediction_tasks{task}{1}, 1);        
    end
    
    psr.num_tasks = num_tasks;
    psr.task_dim = task_dim;
    psr.future_win = future_win;
    psr.past_win = past_win;
    psr.f0 = mean(all_past, 2);
    
    % Filtering/Prediction function handles
    psr.filter = @rff_hsepsr_filter;
    psr.predict = @rff_hsepsr_predict;
    psr.predict_task = @rff_hsepsr_predict_task;
    psr.test = @rff_hsepsr_test;
    
    psr.filter(psr, psr.f0, all_obs(:,1), all_act(:,1));
    psr.predict(psr, psr.f0, all_act(:,1));
    psr.test(psr, psr.f0, all_test_a(:,1));
end

function o = rff_hsepsr_predict(psr, f, a)
    d_o = psr.d_o_;
    d_a = psr.d_a_;

    f_obs = f(1:d_o * psr.past_win);
    f_act = f(d_o*psr.past_win+1:end);
    f_obs = discount(f_obs, d_o, psr.discount_);
    f_act = discount(f_act, d_a, psr.discount_);
    f = [f_obs; f_act];
    
    input = [f; a];
    in_rff = psr.U_1sp_' * func_rff(psr.V_1sp_, input);
    o = psr.Wp_ * in_rff;
end

function o = rff_hsepsr_predict_task(psr, f, a, task_id)
    d_o = psr.d_o_;
    d_a = psr.d_a_;

    f_obs = f(1:d_o * psr.past_win);
    f_act = f(d_o*psr.past_win+1:end);
    f_obs = discount(f_obs, d_o, psr.discount_);
    f_act = discount(f_act, d_a, psr.discount_);
    f = [f_obs; f_act];
    
    input = [f; a];
    in_rff = psr.U_1sp_' * func_rff(psr.V_1sp_, input);
    o = psr.Wtsk_{task_id} * in_rff;
end

function o = rff_hsepsr_test(psr, f, a)
    d_o = psr.d_o_;
    d_a = psr.d_a_;

    a = reshape(a,[],1);
    f_obs = f(1:d_o * psr.past_win);
    f_act = f(d_o*psr.past_win+1:end);
    f_obs = discount(f_obs, d_o, psr.discount_);
    f_act = discount(f_act, d_a, psr.discount_);
    f = [f_obs; f_act];
    
    input = [f; a];
    in_rff = psr.U_tp_' * func_rff(psr.V_tp_, input);    
    o = psr.Wt_ * in_rff;
    o = reshape(o, psr.d_o_, []);
end

function sf = rff_hsepsr_filter(psr, f, o, a)
    d_o = psr.d_o_;
    d_a = psr.d_a_;

    f_obs = f(1:d_o*psr.past_win);
    f_act = f(d_o*psr.past_win+1:end);
    sfo = [f_obs(d_o+1:end); o];
    sfa = [f_act(d_a+1:end); a];        
    sf = [sfo; sfa];
end

function Y = discount(X, d, lambda)
    l = (size(X,1) / d);
    Y = X;
    
    for i = 1:l-1
        Y((i-1)*d+1:i*d,:) = X((i-1)*d+1:i*d,:) * lambda ^ (l-i);        
    end
end
