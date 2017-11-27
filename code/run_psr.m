function [ est_obs, states ] = run_psr( psr, obs, act, initial_state)
if nargin < 4 || isempty(initial_state); initial_state = psr.f0; end;

N = size(obs,2);
d = size(obs,1);
k = psr.future_win;

f = initial_state;
states = zeros(size(initial_state,1),N-k+1);
est_obs = zeros(d,k,N-k+1);

% if horizon == 1
%     % Use 1-step predictor
%     for i=1:N
%         if task == 0 % Predict observation
%             est_obs(:,i) = psr.predict(psr, f, act(:,i));
%         else
%             est_obs(:,i) = psr.predict_task(psr, f, act(:,i), task);
%         end
%             
%         f = psr.filter(psr, f, obs(:,i), act(:,i));
%         states(:,i) = f;
%     end
% else
    
for i=1:N-k+1
    est_obs(:,:,i) = psr.test(psr, f, act(:,i:i+k-1));    
    f = psr.filter(psr, f, obs(:,i), act(:,i));
    states(:,i) = f;
end

a=1;