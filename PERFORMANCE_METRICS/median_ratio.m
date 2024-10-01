function s_bg_ratio = median_ratio(sig,bg)
% get two images, signal and background and calculate the ratio of the
% signal median to the background standard diviation

s_bg_ratio = max(median(sig(:)))/std(bg(:));