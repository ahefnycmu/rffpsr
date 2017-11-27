function F = finite_future_feature_extractor(future_length, extra_feats, lag)
%FINITE_FUTURE_FEATURE_EXTRACTOR   Creates a feature extractor that extracts a window of future observations.
%An observation that exceeds sequence boundary is set to 0. The window is
%reshaped into a single vector (column-order).
%Paramaters:
% lag - Future window at time t starts at t+lag (If not specified, lag is set to 0). 
% future_length - Length of the window
% extra_feats - If true, two extra dimensions are added to each observation
% in the window. The first one is always set to 0. The second is set to 1
% if the corresponding observation exceeds sequence boundary.
    if nargin < 3 || isempty(lag); lag = 0; end
    F = timewin_feature_extractor(future_length, lag, extra_feats);
end
