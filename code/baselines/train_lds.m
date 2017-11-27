function model = train_lds(obs, act, future_win, past_win, options)

if nargin < 5 || isempty(options); options = struct; end;
if ~isfield(options, 'nonblind'); options.nonblind = 0; end;
if ~isfield(options, 'p'); options.p = 50; end;
if ~isfield(options, 'choose_p'); options.choose_p = 0; end;

fut = future_win;
past = past_win;

num_series = length(obs);
data = iddata(obs{1}', act{1}', 1);

for i = 2:num_series
    data = merge(data, iddata(obs{i}', act{i}', 1));
end

alg = 'auto';
if options.nonblind; alg = 'SSARX'; end;
opt = n4sidOptions('N4Weight', alg, 'N4Horizon', [fut past past]);
if(options.choose_p); p = 1:options.p; else p = options.p; end;
model = n4sid(data, p, opt);

