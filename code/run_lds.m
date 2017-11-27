function [ est_obs ] = run_lds(lds, obs, act, horizon)
if nargin < 4 || isempty(horizon); horizon = 1; end;
data = iddata(obs', act', 1);

[d,N] = size(obs);
est_obs = zeros(d,horizon,N);

for k=1:horizon
    oh = predict(lds, data, k);
    est_obs(:,k,:) = reshape(oh.OutputData',d,1,N);
end

