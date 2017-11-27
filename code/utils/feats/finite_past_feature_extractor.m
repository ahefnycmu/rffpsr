function F = finite_past_feature_extractor(past_length, extra_feats, lag) 
%FINITE_PAST_FEATURE_EXTRACTOR   Creates a feature extractor that extracts a window of past observations.
%An observation that exceeds sequence boundary is set to 0. The window is
%reshaped into a single vector (column-order).
%Paramaters:
% lag - Past window at time t ends at t-lag (If not specified, lag is set to 1).
% past_length - Length of the window
% extra_feats - If true, two extra dimensions are added to each observation
% in the window. The second one is always set to 0. The first is set to 1
% if the corresponding observation exceeds sequence boundary.
    if nargin < 3 || isempty(lag); lag = 1; end
    F = timewin_feature_extractor(past_length, -past_length-lag+1, extra_feats);
end
