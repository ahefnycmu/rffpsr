rng(0);

%% Load Data
load ../data/synth.mat
X_trv = [X_tr X_val];
U_trv = [U_tr U_val];

%% Train Models
evaluate_hsepsr = 0; % Set to 0 to avoid training and evaluating HSEPSR

past = 20;
fut = 10;
options = struct;
options.lambda = 1e-3;
options.D = 5000;
options.p = 20;
 
options.gpu_level = 0;
options.reg_maxit = 1000;
options.num_rnd = 1;
 
psr = {}; names = {}; 
runtimes = {};
 
disp('  ***** Training RFF-PSR-joint ***** ');
options.refine = 100;
options.rstep = 0.01;
options.min_rstep = 1e-5;
options.const = 1;
options.val_obs = X_val;
options.val_act = U_val;
options.val_batch = 5;
options.s1_method = 'joint';
options.kernel = 'rbf';
options.D = 20;
s = tic;
names{1} = 'RFF-PSR-joint-refine';
names{2} = 'RFF-PSR-joint';
rng(0);
[ref_rffpsr, rffpsr] = train_rffpsr(X_tr, U_tr, fut, past, {}, options);
psr{1} = ref_rffpsr; runtimes{1} = toc(s);
psr{2} = rffpsr; runtimes{2} = toc(s);

disp('  ***** Training RFF-PSR-cond ***** ');
options.refine = 100;
options.rstep = 0.05;
options.min_rstep = 1e-5;
options.const = 1;
options.val_obs = X_val;
options.val_act = U_val;
options.val_batch = 5;
options.s1_method = 'cond';
s = tic;
names{3} = 'RFF-PSR-cond-refine';
names{4} = 'RFF-PSR-cond';
rng(0);
[ref_rffpsr, rffpsr] = train_rffpsr(X_tr, U_tr, fut, past, {}, options);
psr{3} = ref_rffpsr; runtimes{3} = toc(s);
psr{4} = rffpsr; runtimes{4} = toc(s); 

disp('  ***** Training RFF-ARX ***** ');
options = struct;
options.lambda = 1e-3;
options.D = 5000;
names{5} = 'ARX';
rng(0);
options.discount = 1;
options.p = 20;
s = tic;
psr{5} = train_rff_ar(X_trv, U_trv, fut, past, {}, options);
runtimes{5} = toc(s);

names{6} = 'HSE-PSR';
if evaluate_hsepsr
    options.lambda =5e-4;
    s = tic;
    psr{6} = train_hsepsr(X_tr, U_tr, fut, past, options);
    runtimes{6} = toc(s);
end
 
names{7} = 'Last';
psr{7} = last_obs_predictor(X_tr, fut, options);
runtimes{7} = 0;

disp('  ***** Training LDS ***** ');
options.choose_p = 0;
options.p = 5; %'best';
s = tic;
lds = train_lds(X_trv, U_trv, fut, past, options);
runtimes{8} = toc(s);

%% Evaluate MSE
num_models = length(psr);
models_to_eval = 1:num_models;

if ~exist('mse', 'var') || sum(size(mse) ~= [2*N_tst, num_models+2, fut]) > 0 
    mse = zeros(2*N_tst, num_models+2, fut);
end

all_obs = flatten_features(X_tr, @(x,t) x(:,t), []);
o_max = max(all_obs, [], 2);
o_min = min(all_obs, [], 2);

eval_start = fut;
for i = 1:2*N_tst
    fprintf('Evaluating trajectory %d\n', i);
    obs_test = X_tst{i};
    act_test = U_tst{i};
    
    for j = 1:num_models
        if evaluate_hsepsr || ~strcmp(names{j}, 'HSE-PSR')        
            [obs_h, fh] = run_psr(psr{j}, obs_test, act_test);           
            obs_h = bsxfun(@max, obs_h, o_min);
            obs_h = bsxfun(@min, obs_h, o_max);

            for k = 1:fut
                obs_h_k = reshape(obs_h(:,k,fut:end),size(obs_test,1),[]);
                obs_ref_k = obs_test(:,fut+k-1:end-fut+k);
                mse(i,j,k) = mean(sum((obs_h_k - obs_ref_k).^2, 1));            
            end                
        end
    end    
    
    obs_h = run_lds(lds, obs_test, act_test, fut);
    obs_h = bsxfun(@max, obs_h, o_min);
    obs_h = bsxfun(@min, obs_h, o_max);    
    
    for k=1:fut
        obs_h_k = reshape(obs_h(:,k,fut+k-1:end-fut+k),size(obs_test,1),[]);
        obs_ref_k = obs_test(:,fut+k-1:end-fut+k);
        mse(i,end-1,k) = mean(sum((obs_h_k - obs_ref_k).^2, 1));        
        mse(i,end,k) = sum(var(obs_test(1:end), [], 2));                
    end
end

%% Plot Results
plot(squeeze(mean(mse(:,1,:),1)), 'r*-');
hold on;
plot(squeeze(mean(mse(:,2,:),1)), 'r+--');
plot(squeeze(mean(mse(:,3,:),1)), 'b*-');
plot(squeeze(mean(mse(:,4,:),1)), 'b+--');
plot(squeeze(mean(mse(:,5,:),1)), 'k*-');
plot(squeeze(mean(mse(:,6,:),1)), 'm*-');
plot(squeeze(mean(mse(:,7,:),1)), 'k--');
plot(squeeze(mean(mse(:,8,:),1)), 'm+--');
hold off;
xlabel('Prediction Horizon')
ylabel('Mean Square Error')
legend({'RFF-PSR Joint S1 + Refinement', 'RFF-PSR Joint S1', 'RFF-PSR Conditional S1 + Refinement', ...
    'RFF-PSR Conditional S1', 'ARX', 'HSE-PSR', 'Last Observation', 'N4SID'}, 'Location', 'NorthOutside');
ylim([0.0 0.3]);

