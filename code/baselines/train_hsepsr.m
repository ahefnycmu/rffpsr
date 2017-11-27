function hsepsr = train_hsepsr(obs, act, future_win, past_win, options)
    if nargin < 5 || isempty(options); options = struct; end;
    if ~isfield(options, 'lambda'); options.lambda = 1e-3; end;
    if ~isfield(options, 'range'); options.range = [-past_win -future_win]; end; 
        
    range = options.range;                 
    lambda = options.lambda;
    
    future_feats = finite_future_feature_extractor(future_win, false);
    shifted_future_feats = finite_future_feature_extractor(future_win, false,1);    
    past_feats = finite_past_feature_extractor(past_win, false);
    shifted_past_feats = finite_past_feature_extractor(past_win, false, 0);
    
    all_future_obs = flatten_features(obs, future_feats, range);
    N = size(all_future_obs, 2);    
    all_shifted_obs = flatten_features(obs, shifted_future_feats, range);        
    all_obs = flatten_features(obs, @(X,t) X(:,t), range);            
    
    all_future_act = flatten_features(act, future_feats, range);
    all_shifted_act = flatten_features(act, shifted_future_feats, range);        
    all_act = flatten_features(act, @(X,t) X(:,t), range);    
            
    all_past_obs = flatten_features(obs, past_feats, range);
    all_past_act = flatten_features(act, past_feats, range);
    all_past = [all_past_obs; all_past_act];
    
    all_shifted_past_obs = flatten_features(obs, shifted_past_feats, range);
    all_shifted_past_act = flatten_features(act, shifted_past_feats, range);
    all_shifted_past = [all_shifted_past_obs; all_shifted_past_act];
                
    %% Compute kernel bandwidths using median trick
    s_past = median_bandwidth(all_past, 5000);    
    s_fut_o = median_bandwidth(all_future_obs, 5000);
    s_fut_a = median_bandwidth(all_future_act, 5000);    
    s_o = median_bandwidth(all_obs, 5000);
    s_a = median_bandwidth(all_act, 5000);
    s_past = 4e-0;
    s_fut_o = 2e-0;
    s_fut_a = 2e-0;
    s_o = 1e0;
    s_a = 1e0;
    
    %gram_matrix_rbf = @(X,Y,s) X'*Y; % For debugging
    
    %% Compute Gram Matrices
    G_h = gram_matrix_rbf(all_past, all_past, s_past);  
    G_hsh = gram_matrix_rbf(all_past, all_shifted_past, s_past);  
    G_o = gram_matrix_rbf(all_obs, all_obs, s_o);            
    G_a = gram_matrix_rbf(all_act, all_act, s_a);
    
    G_fo = gram_matrix_rbf(all_future_obs, all_future_obs, s_fut_o);        
    G_fa = gram_matrix_rbf(all_future_act, all_future_act, s_fut_a);    
                
    %% S1 Regression
    R = lambda*N*speye(N);
    Ah = (G_h + R) \ G_h;
    Ash = (G_h + R) \ G_hsh;
   
    % Compaute State gram matrices
    % This is an N^3 storage, N^4 runtime method    
    GGo = zeros(N*N,N);
    GGa = zeros(N*N,N);    
    %GGa_sh = zeros(N*N,N);
    Gah = zeros(N,N,N);
    for i=1:N
        disp(i);
        ai = Ah(:,i);
        Gai = (bsxfun(@times, ai, G_fa) + R) \ diag(ai);
        GGa(:,i) = reshape(Gai * G_fa, [], 1);  
        GGo(:,i) = reshape(G_fo * Gai, [], 1);  
        Gah(:,:,i) = Gai;
        
        %ai = Ash(:,i);
        %Gai = (bsxfun(@times, ai, G_fa) + R) \ diag(ai);
        %GGa_sh(:,i) = reshape(Gai * G_fa, [], 1);  
        %GGo_sh(:,i) = reshape(G_fo * Gai, [], 1);   
    end
    
    G_s = GGa'*GGo;  % state * state  
        
    % Shifted State
    for i=1:N
        fprintf('%d - shifted\n', i)                
        ai = Ash(:,i);
        Gai = (bsxfun(@times, ai, G_fa) + R) \ diag(ai);
        %GGa_sh(:,i) = reshape(Gai * G_fa, [], 1);  
        GGo(:,i) = reshape(G_fo * Gai, [], 1);   
    end
    
    G_ss = GGa'*GGo; % state * shifted state       
    
    %% S2 Regression
    
    % This a matrix that accepts a vector of weights of shifted futures
    % (i.e. the outcome of the previous filtering step)
    % and produces a vector of weights of extended futures.
    W2 = (G_s + lambda * N * speye(N)) \ G_ss;
    
    hsepsr.W2 = W2;
    hsepsr.G_a = G_a;
    hsepsr.G_o = G_o;
    hsepsr.G_fa = G_fa;
    hsepsr.G_fo = G_fo;
    hsepsr.Gah = Gah;
    hsepsr.Ah = Ah;
        
    hsepsr.all_act = all_act;
    hsepsr.all_obs = all_obs;
    hsepsr.all_future_act = all_future_act;
    hsepsr.all_future_obs = all_future_obs;
    hsepsr.s_a = s_a;
    hsepsr.s_fut_a = s_fut_a;
    hsepsr.s_o = s_o;
    
    hsepsr.f0 = ones(N,1)/N;
    %hsepsr.f0 = [1; zeros(N-1,1)];
    hsepsr.future_win = future_win;    

    hsepsr.lambda = lambda;
    
    hsepsr.filter = @hsepsr_filter;
    hsepsr.test = @hsepsr_test;
end
    
function q_a = hsepsr_filter_action(hsepsr, q, a)
    all_act = hsepsr.all_act;                       
    s_a = hsepsr.s_a;        
    N = size(all_act,2);
    G = reshape(reshape(hsepsr.Gah,N*N,N) * q,N,N);
    %G = sum(bsxfun(@times, hsepsr.Gah, reshape(q, 1, 1, [])),3);
        
    k_a = gram_matrix_rbf(all_act, a, s_a);
    q_a = G * k_a;    
end

function qp = hsepsr_filter(hsepsr, q, o, a)
    W2 = hsepsr.W2;    
    G_o = hsepsr.G_o;
    Ah = hsepsr.Ah;
    N = size(Ah,1);
    lambda = hsepsr.lambda;
        
    all_obs = hsepsr.all_obs;    
    s_o = hsepsr.s_o;
    
    % Apply S2 regression
    q = W2 * q;
    
    % Filter action
    q_a = hsepsr_filter_action(hsepsr, q, a);
    
    % Filter observations        
    G = bsxfun(@times, q_a, G_o);
    k_o = gram_matrix_rbf(all_obs, o, s_o);
    qp = (G + lambda * N * speye(N)) \ (q_a .* k_o);
end

function q_fa = hsepsr_test_cond(hsepsr, q, fa)
    fa = reshape(fa,[],1);
    all_future_act = hsepsr.all_future_act;                             
    s_fa = hsepsr.s_fut_a;        
    %G = sum(bsxfun(@times, hsepsr.Gah, reshape(q, 1, 1, [])),3);    
    N = size(all_future_act,2);
    G = reshape(reshape(hsepsr.Gah,N*N,N) * q,N,N);
    
    k_a = gram_matrix_rbf(all_future_act, fa, s_fa);
    q_fa = G * k_a;         
end

function o = hsepsr_test(hsepsr, q, a)
    W2 = hsepsr.W2;    
    G_o = hsepsr.G_o;
    Ah = hsepsr.Ah;
    N = size(Ah,1);
        
    all_fut_obs = hsepsr.all_future_obs;    
    s_o = hsepsr.s_o;
    
    % Apply S2 regression
    q = W2 * q;
    
    q_fa = hsepsr_test_cond(hsepsr, q, a);        
    o = all_fut_obs * q_fa;
    o = reshape(o,[],size(a,2));
end