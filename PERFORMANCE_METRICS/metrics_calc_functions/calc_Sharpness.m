function [ASBG] = calc_Sharpness(RateImage_Sig_med,RateImage_BG_med)
Sig_support = sum(exp(-RateImage_Sig_med(:)));
BG_support = sum(exp(-RateImage_BG_med(:)));

ASBG = Sig_support/BG_support;% No good