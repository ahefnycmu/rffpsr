function [psr, non_refined_psr] = train_rffpsr(obs, act, ...
    future_win, past_win, options)
%TRAIN_RFFPSR - Trains an HSE-PSR using random fourier features.
%This function uses 2 stage regression for initialization followed by 
% backpropagation through time (BPTT) refinement.
%
%The 2 stage regression works as follows:
% -S1A: history --> (fut_o | fut_a)
% -S1B: history -----> (ext_fut_o | ext_fut_a)
%                  |-> (oo | a)                  
% -S2: (fut_o | fut_a) ----> (ext_fut_o | ext_fut_a)
%                        |-> (oo | a) 
%This function assumes the data is collected using a blind policy.
%
%Parameters:
%  obs :    A cell array of observation trajectories. Each trajectory is a 
%           dxT matrix, where d is obs dimension and T is trajectory
%           length.
%  act :    A cell array of action trajectories. Each trajectory is a dxT
%           matrix, where d is act dimension and T is trajectory length.
%  future_win:  Future length
%  past_win:    History length
%  options:     Optional parameters structure. See below for details on
%               optional parameters and default values.
%
%Returns:
% PSR after and before BPTT refinement.

    % Default Parameters    
    if nargin < 5 || isempty(options); options = struct; end    
    if ~isfield(options, 'reg_maxit'); options.reg_maxit = 1000; end % Max. iterations for regression optimization algorithms
    if ~isfield(options, 'D'); options.D = 1000; end % Number of RFF feature pairs
    if ~isfield(options, 'p'); options.p = 50; end % Projection dimension
    if ~isfield(options, 'lambda'); options.lambda = 1e-3; end % L2 Regularization          
    if ~isfield(options, 's'); options.s = -1; end % Kernel bandwidth
    if ~isfield(options, 'random'); options.random = 0; end % Random initialization
    if ~isfield(options, 'range'); options.range = [-past_win -future_win]; end
    
    % S1 regression method. Possible choices are 'cond' and 'joint'
    if ~isfield(options, 's1_method'); options.s1_method = 'joint'; end 
    if (~strcmp(options.s1_method, 'joint') && ~strcmp(options.s1_method, 'cond'))
        error('Invalid S1 regression method. Possible choices are "cond" and "joint"');
    end
                    
    % Const feature flags: Determines which representations should
    % include an additional constant features after PCA.
    % LSB to MSB: history, action, test_action, extended_test_action
    if ~isfield(options, 'const'); options.const = 1; end
    
    const_history = bitand(options.const,1);
    const_act = bitand(options.const,2);
    const_test_a = bitand(options.const,4);
    const_ex_test_a = bitand(options.const,8);
    
    % Refinement options
    if ~isfield(options, 'refine'); options.refine = 0; end % Refinement iterations
    if ~isfield(options, 'min_rstep'); options.min_rstep = 1e-6; end % Minimum refinement step
    if ~isfield(options, 'rstep'); options.rstep = 0.1; end % Refinement step        
    if ~isfield(options, 'val_obs'); options.val_obs = {}; end % Validation set observations
    if ~isfield(options, 'val_act'); options.val_act = {}; end % Validation set actions
    
    % Early stopping uses minimum validation error in 'val_batch'
    % iterations to make stopping decision.
    % By default, options.val_batch is set to options.refine to disable 
    % early stopping.
    if ~isfield(options, 'val_batch'); options.val_batch = options.refine; end
    
    range = options.range;
    lambda = options.lambda;
    D = options.D;
    p = options.p;        
    k = future_win;
         
    [all.past_obs, series_index, time_index] = flatten_features(obs, finite_past_feature_extractor(past_win, false), range);
    all.past_act = flatten_features(act, finite_past_feature_extractor(past_win, false), range);
    all.past = [all.past_obs; all.past_act];    
    all.test_o = flatten_features(obs, finite_future_feature_extractor(k, false), range);
    all.test_a = flatten_features(act, finite_future_feature_extractor(k, false), range);    
    all.shtest_o = flatten_features(obs, finite_future_feature_extractor(k, false, 1), range);
    all.shtest_a = flatten_features(act, finite_future_feature_extractor(k, false, 1), range);
               
    d.h = size(all.past, 1);    
    d.o = size(obs{1}, 1);
    d.a = size(act{1}, 1);
    d.to = size(all.test_o,1);
    d.ta = size(all.test_a,1);
    
    all.obs = all.test_o(1:d.o,:);
    all.act = all.test_a(1:d.a,:);        
        
    N = size(all.test_o, 2);
    
    reg_options = struct();
    
    %% Transform training data to feature space    
    if strcmp(options.kernel, 'rbf') || strcmp(options.kernel, 'nst')     
        % Compute kernel bandwidths using median trick
        disp('Estimating Kernel Bandwidths');
        tic;
        if options.s <= 0
            s_h = median_bandwidth(all.past, 5000);            
            s_o = median_bandwidth(all.obs, 5000);
            s_a = median_bandwidth(all.act, 5000);        
            s_to = median_bandwidth(all.test_o, 5000);
            s_ta = median_bandwidth(all.test_a, 5000);
        else
            s_h = options.s;
            s_o = options.s;
            s_a = options.s;
            s_to = options.s;
            s_ta = options.s;
        end
        toc;
    end

    
    % Sample from kernel spectra
    tic;
    disp('Sampling Frequencies');
    V.h = randn(D,d.h) / s_h;    
    V.a = randn(D,d.a) / s_a;
    V.o = randn(D,d.o) / s_o;        
    V.to = randn(D,d.to) / s_to;
    V.ta = randn(D,d.ta) / s_ta; 
    toc;

    % Create feature extractors
    feat.h = @(X_o, X_a) func_rff(V.h, [X_o; X_a]);    
    feat.o = @(X) func_rff(V.o, X);
    feat.a = @(X) func_rff(V.a, X);    
    feat.to = @(X) func_rff(V.to, X);
    feat.ta = @(X) func_rff(V.ta, X); 
              
    disp('Computing low dim representation of history');
    tic;        
    [U.h,~, all.past_feat] = rand_svd_f(@(s,e) feat.h( ...
         all.past_obs(:,s:e), all.past_act(:,s:e)),N,p,[],50);
    
    if const_history
        all.past_feat = [all.past_feat; ones(1,N)];
        prj_feat.h = @(X_o, X_a) prj_add_const(U.h, feat.h(X_o, X_a));
    else
        prj_feat.h = @(X_o, X_a) U.h' * feat.h(X_o, X_a); 
    end
    
    K.h = size(all.past_feat,1);                  
    toc;
            
    disp('Computing low dim representation of observations and actions');
    tic;
    [U.a, ~, all.act_feat] = rand_svd_f(@(s,e) feat.a(all.act(:,s:e)),N,p,[],50);        
    if const_act
        all.act_feat = [all.act_feat; ones(1,N)];                
        prj_feat.a = @(X) prj_add_const(U.a, feat.a(X));
    else
        prj_feat.a = @(X) U.a' * feat.a(X);
    end
        
    K.a = size(all.act_feat,1);
    
    [U.o, ~, all.obs_feat] = rand_svd_f(@(s,e) feat.o(all.obs(:,s:e)),N,p,[],50);        
    K.o = size(all.obs_feat,1); 
    prj_feat.o = @(X) U.o' * feat.o(X);
        
    all.oo = kr_product(all.obs_feat, all.obs_feat);    
    [U.oo, ~, all.oo_feat] = rand_svd_f(@(s,e) all.oo(:,s:e),N,p,[],50);     
    K.oo = size(all.oo_feat,1);        
    toc;
    
    disp('Computing low dim representation of tests');
    tic;
    [U.to, ~, all.test_o_feat] = rand_svd_f(@(s,e) feat.to(all.test_o(:,s:e)),N,p,[],50);    
    [U.ta, ~, all.test_a_feat] = rand_svd_f(@(s,e) feat.ta(all.test_a(:,s:e)),N,p,[],50);
    
    prj_feat.to = @(X) U.to' * feat.to(X);
    if const_test_a
        all.test_a_feat = [all.test_a_feat; ones(1,N)];    
        prj_feat.ta = @(X) prj_add_const(U.ta, feat.ta(X));
    else
        prj_feat.ta = @(X) U.ta' * feat.ta(X);
    end
              
    K.to = size(all.test_o_feat,1);
    K.ta = size(all.test_a_feat,1);
    
    % Shifted tests
    all.shtest_o_feat = blk_func(@(s,e) U.to' * feat.to(all.shtest_o(:,s:e)),N); 
    all.shtest_a_feat = blk_func(@(s,e) prj_feat.ta(all.shtest_a(:,s:e)),N);     
    
    % Extended tests
    all.extest_a = kr_product(all.act_feat, all.shtest_a_feat);    
    [U.eta, ~, all.extest_a_feat] = rand_svd_f(@(s,e) all.extest_a(:,s:e),N,p,[],50);
    
    if const_ex_test_a
        assert(const_test_a && const_act);
        
        % The KR product of action and shifted test actions already
        % contains a constant feature. Make sure it stays there.
        U.eta = orth([U.eta [zeros(size(U.eta,1)-1,1); 1]]);
        all.extest_a_feat = U.eta' * all.extest_a;
    end
    
    K.eta = size(all.extest_a_feat, 1);
    
    % Note that current observation is the "lower order" factor.
    % This makes filtering easier.
    all.extest_o = kr_product(all.shtest_o_feat, all.obs_feat); 
    [U.eto, ~, all.extest_o_feat] = rand_svd_f(@(s,e) all.extest_o(:,s:e),N,p,[],50);
    K.eto = size(all.extest_o_feat, 1);
    toc;
    
    disp('Computing inverse feature maps');
    tic;
    W.rff2obs = ridge_regression(all.obs_feat,all.obs,lambda);
    W.oo2obs = ridge_regression(all.oo_feat,all.obs,lambda);
    W.oo2rffobs = ridge_regression(all.oo_feat,all.obs_feat,lambda);
    W.rff2to = ridge_regression(all.test_o_feat,all.test_o,lambda);
    toc;
                         
    if strcmp(options.s1_method, 'joint')   
        %% S1 Regression
        disp('S1 Regression');
        tic;
        s1_in = all.past_feat;
                                
        d_s1_out(1) = size(all.test_a_feat,1) * size(all.test_o_feat,1);
        d_s1_out(2) = size(all.test_a_feat,1) * size(all.test_a_feat,1);
        d_s1_out(3) = size(all.extest_a_feat,1) * size(all.extest_o_feat,1);
        d_s1_out(4) = size(all.extest_a_feat,1) * size(all.extest_a_feat,1);
        d_s1_out(5) = size(all.act_feat,1) * size(all.oo_feat,1);
        d_s1_out(6) = size(all.act_feat,1) * size(all.act_feat,1);
        
        ds1 = cumsum(d_s1_out);
        s1_out_all = zeros(ds1(end), N);
        
        % State
        s1_out_all(1:ds1(1), :) = kr_product(all.test_a_feat, all.test_o_feat);        
        s1_out_all(ds1(1)+1:ds1(2), :) = kr_product(all.test_a_feat, all.test_a_feat);

        % Extended State
        s1_out_all(ds1(2)+1:ds1(3), :) = kr_product(all.extest_a_feat, all.extest_o_feat);
        s1_out_all(ds1(3)+1:ds1(4), :) = kr_product(all.extest_a_feat, all.extest_a_feat);

        % Immediate Prediction
        s1_out_all(ds1(4)+1:ds1(5), :) = kr_product(all.act_feat, all.oo_feat);
        s1_out_all(ds1(5)+1:ds1(6), :) = kr_product(all.act_feat, all.act_feat);            
        
        W.s1 = ridge_regression(s1_in, s1_out_all, lambda);
        clear s1_out_all;
        toc;

        %% Compute States
        disp('Computing States');
        tic;

        est_s1_out = W.s1 * s1_in;

        states = zeros(K.ta * K.to, N);        
        % Extended and immediate states
        L = K.eta * K.eto;
        exim_states = zeros(L + K.a * K.oo, N);                
        
        for i = 1:N        
            C_tota = reshape(est_s1_out(1:ds1(1),i), K.to, K.ta);
            C_tata = reshape(est_s1_out(ds1(1)+1:ds1(2),i), K.ta, K.ta);
            C_to_ta = reg_divide(C_tota * C_tata, C_tata * C_tata, lambda);         
            states(:,i) = reshape(C_to_ta, [], 1);

            C_etoeta = reshape(est_s1_out(ds1(2)+1:ds1(3),i), K.eto, K.eta);
            C_etaeta = reshape(est_s1_out(ds1(3)+1:ds1(4),i), K.eta, K.eta);
            C_eto_eta = reg_divide(C_etoeta * C_etaeta, C_etaeta * C_etaeta, lambda);         
            exim_states(1:L,i) = reshape(C_eto_eta, [], 1);

            C_ooa = reshape(est_s1_out(ds1(4)+1:ds1(5),i), K.oo, K.a);
            C_aa = reshape(est_s1_out(ds1(5)+1:ds1(6),i), K.a, K.a);
            C_oo_a = reg_divide(C_ooa * C_aa, C_aa * C_aa, lambda);         
            exim_states(L+1:end,i) = reshape(C_oo_a, [], 1);
        end  
        clear est_s1_out;
        toc;                 

        %% Project States
        disp('Project states');
        tic;
        [U.st,~,states] = rand_svd_f(@(s,e) states(:,s:e),N,p,1,50);                
        K.s = size(states, 1);
        toc;
        
        %% S2 Regression
        disp('S2 Regression');
        tic;
        s2_in = states;
        s2_out_all = exim_states;
        %W.s2 = cg_ridge(s2_in, s2_out_all, lambda, reg_options);
        W.s2 = ridge_regression(s2_in, s2_out_all, lambda);
        
        W.s2_ex = W.s2(1:L,:);    
        W.s2_oo = W.s2(L+1:end,:);                   
        toc;
    else % strcmp(options.s1_method, 'cond')
        %% S1A Regression
        disp('S1A Regression');
        tic;
        s1a_in = kr_product(all.past_feat,all.test_a_feat);
        s1a_out = all.test_o_feat;           
        reg_options.maxit = options.reg_maxit;        
        W.s1a = ridge_regression(s1a_in, s1a_out, lambda);
        W.s1a = reshape(W.s1a, [], K.h);
        toc; 

        %% S1B Regression
        disp('S1B Regression');
        tic;
        s1b_in = kr_product(all.past_feat,all.extest_a_feat);        
        s1b_out = all.extest_o_feat;                
        W.s1b = ridge_regression(s1b_in, s1b_out, lambda);
        W.s1b = reshape(W.s1b, [], K.h);
        toc; 

        %% Project States
        disp('Project states');
        tic;
        states = W.s1a * all.past_feat;
        [U.st,~,states] = rand_svd_f(@(s,e) states(:,s:e),N,p,1,50);            
        K.s = size(states, 1);

        ex_states = W.s1b * all.past_feat;                    
        toc;
        
        %% S2 Regression
        disp('S2 Regression - Obs Covariance');    
        tic;
        s2_oo_in = kr_product(states, all.act_feat);                
        s2_oo_out = all.oo_feat;
        W.s2_oo = cg_ridge(s2_oo_in, s2_oo_out, lambda, reg_options);     
        W.s2_oo = reshape(W.s2_oo, [], K.s);                        
        toc;

        disp('S2 Regression - Extended future');    
        tic;
        s2_ex_in = states;                
        s2_ex_out = ex_states;
        W.s2_ex = cg_ridge(s2_ex_in, s2_ex_out, lambda, reg_options);                    
        toc;
    end
    
    W.s2_obs = W.oo2obs * reshape(W.s2_oo, K.oo, []);
    W.s2_rffobs = W.oo2rffobs * reshape(W.s2_oo, K.oo, []);
        
    % Compute a linear map from history to projected states. This is 
    % helpful for initializing validation states.
    W.s1a_proj = ridge_regression(all.past_feat, states, lambda);
    
%      %% S2 Regression
%     disp('S2 Regression - Obs Covariance');    
%     tic;
%     s2_oo_in = kr_product(states, all.act_feat);                
%     s2_oo_out = all.oo_feat;
%     W.s2_oo = cg_ridge(s2_oo_in, s2_oo_out, lambda, reg_options);     
%     W.s2_oo = reshape(W.s2_oo, [], K.s);                        
%     toc;
% 
%     disp('S2 Regression - Extended future');    
%     tic;
%     s2_ex_in = states;                
%     s2_ex_out = ex_states;
%     W.s2_ex = cg_ridge(s2_ex_in, s2_ex_out, lambda, reg_options);                    
%     toc;
    
    %% Replace initialized states and parameters with random values
    if options.random
        rng(options.random);
        W.s2_ex = randn(size(W.s2_ex));
        W.s2_oo = randn(size(W.s2_oo));
        states = randn(size(states));
    end

    %% Build Model
    if options.refine > 0
        non_refined_psr = build_model(future_win, states, all, W, U, d, K, feat, prj_feat, ...
            lambda, reg_options);
    
        W.s2_h = non_refined_psr.W_.s2_h; 
    end            
                
    %% Refinement       
    use_validation = options.refine > 0 && ~isempty(options.val_obs);
        
    if use_validation
        disp('Refinement - Validation Features');        
        tic;
        [val_all.past_obs, val_series_index] = flatten_features(options.val_obs, finite_past_feature_extractor(past_win, false), range);
        val_all.past_act = flatten_features(options.val_act, finite_past_feature_extractor(past_win, false), range);        
        val_n = size(val_all.past_obs, 2);
        val_all.obs = flatten_features(options.val_obs, @(X,t) X(:,t), range);
        val_all.act = flatten_features(options.val_act, @(X,t) X(:,t), range);    
        val_all.test_o = flatten_features(options.val_obs, finite_future_feature_extractor(k, false), range);
        val_all.test_a = flatten_features(options.val_act, finite_future_feature_extractor(k, false), range);    
        val_all.shtest_o = flatten_features(options.val_obs, finite_future_feature_extractor(k, false, 1), range);
        val_all.shtest_a = flatten_features(options.val_act, finite_future_feature_extractor(k, false, 1), range);
                
        
        val_all.past_feat = blk_func(@(s,e) prj_feat.h(val_all.past_obs(:,s:e), val_all.past_act(:,s:e)), val_n);
                
        val_all.obs_feat = blk_func(@(s,e) U.o' * feat.o(val_all.obs(:,s:e)), val_n);
        val_all.act_feat = blk_func(@(s,e) prj_feat.a(val_all.act(:,s:e)), val_n);
        val_all.oo_feat = blk_func(@(s,e) U.oo' * kr_product(val_all.obs_feat(:,s:e), val_all.obs_feat(:,s:e)), val_n);
                        
        val_all.test_o_feat = blk_func(@(s,e) U.to' * feat.to(val_all.test_o(:,s:e)),val_n);
        val_all.test_a_feat = blk_func(@(s,e) prj_feat.ta(val_all.test_a(:,s:e)),val_n);
        val_all.shtest_o_feat = blk_func(@(s,e) U.to' * feat.to(val_all.shtest_o(:,s:e)),val_n);
        val_all.shtest_a_feat = blk_func(@(s,e) prj_feat.ta(val_all.shtest_a(:,s:e)),val_n);
    
        val_all.extest_a_feat = blk_func(@(s,e) U.eta' * kr_product(val_all.act_feat(:,s:e), val_all.shtest_a_feat(:,s:e)), val_n);        
        
        % Note that current observation is the "lower order" factor.
        % This makes filtering easier.
        val_all.extest_o_feat = blk_func(@(s,e) U.eto' * kr_product(val_all.shtest_o_feat(:,s:e), val_all.obs_feat(:,s:e)), val_n);                
        toc;                
        
        disp('Refinement - Initial Validation Error');
        tic;
        % Compute validation states
        val_states = W.s1a_proj * val_all.past_feat;
        val_states = filter_trajs( ...
            val_series_index, val_states, val_all, W, U, K, lambda);
                
        init_error = validation_error(val_states, val_all, W, K, U);
        best_val_error = init_error;
        best_val_iteration = 0;
        last_batch_val_error = Inf;
        batch_val_error = Inf;

        best_W = W;                
        toc;
        
        fprintf('Initial Validation error = %f\n', best_val_error);
    elseif options.refine > 0
        % Compute states
        states = W.s1a_proj * all.past_feat;
        states = filter_trajs( ...
            series_index, states, all, W, U, K, lambda);
                
        init_error = validation_error(states, all, W, K, U);
        fprintf('Initial Error = %f\n', init_error);
    end
    
    rstep = options.rstep;    
        
    i = 1;
    while i <= options.refine                                
        fprintf('Refinement - Round %d of %d\n', i, options.refine);
        tic;
        
        num_trajs = max(series_index);
        
        for t=1:num_trajs   
            traj_idx = find(series_index == t);            
                              
            traj = struct('obs', all.obs(:,traj_idx), ...
                'to', all.test_o(:,traj_idx), ...
                'o_feat', all.obs_feat(:,traj_idx), ...
                'oo_feat', all.oo_feat(:,traj_idx), ...
                'a_feat', all.act_feat(:,traj_idx), ...
                'to_feat', all.test_o_feat(:,traj_idx), ...
                'ta_feat', all.test_a_feat(:,traj_idx), ...
                'eto_feat', all.extest_o_feat(:,traj_idx), ...
                'eta_feat', all.extest_a_feat(:,traj_idx));                                
            
            % Update states (Feed forward)  
            traj_f0 = W.s1a_proj * all.past_feat(:,traj_idx(1));
            [traj_states,v_all,C_oo_prj_all,C_eto_ta_all,A_all,B_all] = ...
                filter_traj(t, series_index, traj_f0, all, W, U, K, lambda);
        
            % Back propagation            
            [g_Wex, g_Woo, g_Wh] = bp_traj(traj_states,v_all, ...
                C_oo_prj_all,C_eto_ta_all,A_all,B_all,traj ...
                , W, U, K, lambda);
                       
            W.s2_ex = W.s2_ex - rstep * g_Wex;
            W.s2_oo = W.s2_oo - rstep * g_Woo;
            W.s2_h = W.s2_h - rstep * g_Wh;
        end     
                                                    
        if use_validation                         
            val_states = filter_trajs(val_series_index, val_states, ...
                val_all, W, U, K, lambda);
            
            val_error = validation_error(val_states, val_all, W, K, U);                                                     
            fprintf('Validation error = %f\n', val_error);
            
            if val_error < best_val_error
                best_W = W;
                best_val_iteration = i;
                best_val_error = val_error;
            end
            
            batch_val_error = min(val_error, batch_val_error);
            
            if mod(i, options.val_batch) == 0
                eps = 1e-3;
                % End of validation batch. Check for early stopping.
                if batch_val_error > (1+eps) * last_batch_val_error
                    % Large increase in error
                    % Try decreasing step size
                    rstep = rstep / 2;
                    if rstep < options.min_rstep                
                        fprintf('Early stopping after %d iterations\n', i);
                        break;
                    else
                        fprintf('Reduced step size to %e at iteration %d\n', rstep, i);
                        i = i - options.val_batch;                        
                        W = last_best_W;
                    end
                elseif batch_val_error > (1-eps) * last_batch_val_error
                    % Small change in error. Stop
                    fprintf('Early stopping after %d iterations\n', i);
                    break;
                else
                    % Probably can still improve, proceed.
                    last_batch_val_error = batch_val_error;
                    last_best_W = best_W;
                    batch_val_error = Inf;                    
                end                                
            end
        else
            % Compute states            
            states = filter_trajs( ...
                series_index, states, all, W, U, K, lambda);

            tr_error = validation_error(states, all, W, K, U);
            fprintf('Error = %f\n', tr_error);
        end
        
        i = i + 1;
        toc;
    end    
    
    if use_validation
        W = best_W;
        fprintf('Finished refinement in %d iterations\n', i-1);
        fprintf('Using weights from iteration %d\n', best_val_iteration);
        fprintf('Validation Error: start=%f end=%f\n', init_error, last_batch_val_error);
    elseif options.refine > 0
        fprintf('Error: start=%f end=%f\n', init_error, tr_error);
    end
         
    states = filter_trajs(series_index, states, all, W, U, K, lambda);
    
    psr = build_model(future_win, states, all, W, U, d, K, feat, prj_feat, ...
        lambda, reg_options);
    
    if options.refine == 0
        non_refined_psr = psr;
    end
end

function val_err = validation_error(val_states, val_all, W, K, U)
    % Horizon Error
    val_rf_to_in = kr_product(val_states, val_all.test_a_feat);
    val_rf_to_out = val_all.test_o;        
    val_err = norm(val_rf_to_out - W.s2_h * val_rf_to_in, 'fro');
end

function psr = build_model(future_win, states, all, W, U, d, K, feat, prj_feat, lambda, reg_options)    
    %% Additional Predictors    
    disp('S2 Regression - Horizon Prediction');
    tic;
    s2_h_in = kr_product(states, all.test_a_feat);    
    s2_h_out = all.test_o_feat;
    W.s2_h = ridge_regression(s2_h_in, s2_h_out, lambda);
    W.rff2obs_test = ridge_regression(all.test_o_feat, all.test_o, lambda);    
    W.s2_h = W.rff2obs_test * W.s2_h;
    
    disp('S2 Regression - Additional 1-step Predictors');
    tic;
    s2_1s_in = kr_product(states, all.act_feat);            
    s2_out_obs = all.obs_feat;
                
    s2_1s_out = s2_out_obs;
    W.s2_1s = ridge_regression(s2_1s_in, s2_1s_out, lambda);

    idx=0;                 
    W.s2_rffobs = W.oo2rffobs * reshape(W.s2_oo, K.oo, []);
    idx=idx+K.o;
    
    assert(idx == size(W.s2_1s,1));
    toc;                
            
    %% Build Model
    % Debugging
    psr.states_ = states;
    
    % Private data members
    psr.obs_ = all.obs;
    psr.obs_feat_ = all.obs_feat;
    psr.W_ = W;
        
    psr.U_ = U;             
    psr.K_ = K;    
    psr.d_ = d;        
    
    psr.feat_ = feat;
    psr.prj_feat_ = prj_feat;
    
    psr.lambda_ = lambda;
    
    % Public data memebers
    psr.future_win = future_win;
    psr.f0 = mean(states,2);
            
    % Filtering/Prediction function handles
    psr.state_from_finite_history = @state_from_finite_history;
    psr.filter = @hsepsr_filter;
    psr.predict = @rffpsr_predict;    
    psr.test = @rffpsr_test;
    
    % Test filtering and prediction functions
    psr.filter(psr, psr.f0, all.obs(:,1), all.act(:,1));
    psr.predict(psr, psr.f0, all.act(:,1));
    psr.test(psr, psr.f0, reshape(all.test_a(:,1), d.a, future_win));    
end

function s = state_from_finite_history(psr, past_o, past_a)
    past_o = reshape(past_o, [], 1);
    past_a = reshape(past_a, [], 1);
    s = psr.U_.st' * (psr.W_.s1a * (psr.prj_feat_.h(past_o, past_a)));
end

function o = rffpsr_predict(psr, f, a)
    % Convert action to RFF features
    a_rff = psr.prj_feat_.a(a);
    
    % Convert state to a 1-step predictor and apply to action
    %o = reshape(psr.W_.s2_obs * f, [], psr.K_.a) * a_rff;        
    o = psr.W_.s2_obs * kron(f, a_rff);
end

function o = rffpsr_test(psr, f, a)
    a = reshape(a,[],1);
    a_rff = psr.prj_feat_.ta(a); 
    o = psr.W_.s2_h * rowkron(f', a_rff')';
    o = reshape(o, [], psr.future_win);        
end

function [sf,v,C_oo_prj,C_eto_ta,A,B] = rffpsr_filter_core(f, o_feat, a_feat, W, U, K, lambda)        
    % Obtain v = C_oo \ o_rff
    C_oo_prj = reshape(W.s2_oo * f, [], K.a) * a_feat;
    C_oo = reshape(U.oo * C_oo_prj, K.o, K.o);
            
    v = reg_divide(o_feat' * C_oo, C_oo * C_oo, lambda)';
    
    % Obtain extended state
    C_ex = reshape(W.s2_ex * f, [], K.eta);            
    
    % Condition on action
    B = reshape(reshape(U.eta', [], K.a) * a_feat, K.eta, K.ta);
    C_eto_ta = C_ex * B;
    
    % Multiply by v and project state
    UU = reshape(U.eto, K.o, []);
    A = reshape(v' * UU, K.to, K.eto);
    sf = reshape(A * C_eto_ta, 1, []);
    sf = U.st' * sf';
    
    C_eto_ta = reshape(C_eto_ta, [], 1);
    A = reshape(A, [], 1);
    B = reshape(B, [], 1);
end

function [g_f,g_Wex, g_Woo] = rffpsr_backprop(g_sf, f, o_feat, a_feat, W, U, K, lambda, ...
    v, C_oo_prj, C_eto_ta, A, B)
        A = reshape(A, K.to, K.eto);
        B = reshape(B, K.eta, K.ta);
            
        g_Usf = g_sf * U.st';
        Q = reshape(g_Usf', K.to, [])' * A;
        g_Cex = reshape(Q' * B', 1, K.eto*K.eta);
                        
        g_Wex = rowkron(f', g_Cex);          
        g_f1 = g_Cex * W.s2_ex;
        
        UC_ex = reshape(U.eto * reshape(C_eto_ta,K.eto,K.ta), K.o, []);
        g_v = g_Usf * UC_ex';
        
        C_oo = reshape(U.oo * C_oo_prj, K.o, K.o);        
        C_oo2 = C_oo * C_oo;
        C_oo2(1:K.o+1:end) = C_oo2(1:K.o+1:end) + lambda;
                               
        gviCoo2 = g_v / C_oo2;
        g_Cooprj_1 = -rowkron(v',gviCoo2 * C_oo);
        g_Cooprj_2 = -rowkron((C_oo*v)', gviCoo2);
        g_Cooprj_3 = rowkron(o_feat', gviCoo2);                
        g_Cooprj = (g_Cooprj_1+g_Cooprj_2+g_Cooprj_3) * U.oo;
        
        g_f2 = g_Cooprj * reshape(W.s2_oo, K.oo, []);
        g_f2 = sum(reshape(g_f2 .* repmat(a_feat', 1, K.s), K.a, K.s));
        
        g_f = g_f1 + g_f2;
        g_Woo = rowkron(rowkron(f', a_feat'), g_Cooprj);        
end

function [states,v_all,C_oo_prj_all,C_eto_ta_all,A_all,B_all] = filter_traj(traj, series_index, f0, all, W, U, K, lambda)
    traj_idx = find(series_index == traj);                            
    n = length(traj_idx);
    f = f0;

    states = zeros(K.s,n);
    v_all = zeros(K.o,n);
    C_oo_prj_all = zeros(K.oo,n);
    C_eto_ta_all = zeros(K.eto*K.ta,n);
    A_all = zeros(K.eto*K.to,n);
    B_all = zeros(K.eta*K.ta,n);
              
    for jj = 1:n
        j = traj_idx(jj);
        states(:,jj) = f;
        o_feat = all.obs_feat(:,j);
        a_feat = all.act_feat(:,j);
        [f,v,C_oo_prj,C_eto_ta,A,B] = rffpsr_filter_core(f, o_feat, a_feat, W, U, K, lambda);                    

        v_all(:,jj) = v;
        C_oo_prj_all(:,jj) = C_oo_prj;
        C_eto_ta_all(:,jj) = C_eto_ta;
        A_all(:,jj) = A;
        B_all(:,jj) = B;
    end
end

function [g_Wex, g_Woo, g_Wh] = bp_traj(states, v_all, C_oo_prj_all, ...
    C_eto_ta_all , A_all, B_all, traj, W, U, K, lambda)
    n = size(states,2);
    d_to = size(traj.to,1);     
    
    % Difference between predicted and actual observations        
    diff_all = traj.to - ...
      	W.s2_h * kr_product(states,traj.ta_feat);

    g_Wex = zeros(1,K.s*K.eto*K.eta);
    g_Woo = zeros(1,K.s*K.oo*K.a);
    g_Wh = zeros(1,K.s*d_to*K.ta);
    g_sf = zeros(1,K.s);
        
    a_k_d = kr_product(traj.ta_feat, diff_all);
    
    for i=n-1:-1:1                  
        sf = states(:,i+1);                            
                        
        g_Wh = g_Wh - 2*rowkron(sf',a_k_d(:,i+1)');
        g_sf = g_sf - 2*a_k_d(:,i+1)' * reshape(W.s2_h, [], K.s);
                        
        f = states(:,i);            
        o_feat = traj.o_feat(:,i);
        a_feat = traj.a_feat(:,i);
        v = v_all(:,i);
        C_oo_prj = C_oo_prj_all(:,i);
        C_eto_ta = C_eto_ta_all(:,i); 
        A = A_all(:,i);
        B = B_all(:,i);

        [g_sf,gj_Wex, gj_Woo] = rffpsr_backprop(g_sf, f, o_feat, a_feat, ...
            W, U, K, lambda, v, C_oo_prj, C_eto_ta, A, B);

        g_Woo = g_Woo + gj_Woo;
        g_Wex = g_Wex + gj_Wex;          
    end
        
    g_Woo = (reshape(g_Woo, K.oo*K.a, K.s) + lambda * W.s2_oo) / (n-1);
    g_Wex = (reshape(g_Wex, K.eto*K.eta, K.s) + lambda * W.s2_ex) / (n-1);
    g_Wh = (reshape(g_Wh, d_to, K.s*K.ta) + lambda * W.s2_h) / (n-1);
    
    g_norm = sqrt(norm(g_Woo, 'fro')^2 + norm(g_Wex, 'fro')^2 + norm(g_Wh, 'fro')^2);
    
    %fprintf('=======================================> g_norm=%d\n', g_norm);
    
    max_norm = 10.0;
    if (g_norm > max_norm)
        g_Woo = (g_Woo / g_norm) * max_norm;
        g_Wex = (g_Wex / g_norm) * max_norm;
        g_Wh = (g_Wh / g_norm) * max_norm;
    end
end

function [states,v_all,C_oo_prj_all,C_eto_ta_all,A_all,B_all] = filter_trajs(series_index, states0, all, W, U, K, lambda, bp_inputs)                
    if nargin < 8; bp_inputs = 0; end    
    num_traj = max(series_index);
    states = zeros(size(states0));
    if isa(states0, 'gpuArray'); states = def_gpuArray(states); end
    n = size(states, 2);
    
    if bp_inputs
        v_all = zeros(K.o,n);
        C_oo_prj_all = zeros(K.oo,n);
        C_eto_ta_all = zeros(K.eto*K.ta,n);
        A_all = zeros(K.eto*K.to,n);
        B_all = zeros(K.eta*K.ta,n);
    end
    
    for t = 1:num_traj                
        traj = find(series_index == t);                            
        f = states0(:,traj(1));

        for j = traj
            states(:,j) = f;
            o_feat = all.obs_feat(:,j);
            a_feat = all.act_feat(:,j);
            [f,v,C_oo_prj,C_eto_ta,A,B] = rffpsr_filter_core(f, o_feat, a_feat, W, U, K, lambda);                    
            
            if bp_inputs
                v_all(:,j) = v;
                C_oo_prj_all(:,j) = C_oo_prj;
                C_eto_ta_all(:,j) = C_eto_ta;
                A_all(:,j) = A;
                B_all(:,j) = B;
            end
        end
    end   
end        

function sf = hsepsr_filter(psr, f, o, a)
    % Convert action and observation to features    
    a_feat = psr.prj_feat_.a(a);
    o_feat = psr.U_.o' * psr.feat_.o(o);
    
    sf = rffpsr_filter_core( ...
        f, o_feat, a_feat, psr.W_, psr.U_, psr.K_, psr.lambda_);       
end

function Y = prj_add_const(U, X)
    Y = [U' * X; ones(1, size(X,2))];    
end
