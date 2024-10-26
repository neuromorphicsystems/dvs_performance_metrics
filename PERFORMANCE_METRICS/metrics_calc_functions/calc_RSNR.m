function [RSNR, RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(RateImage_Sig,RateImage_BG,matrix_size)
% get two event rate images: one for signal, and one for background.
% calculate the Rate-Signal-to-Noise ratio for the data.
% 
init_mat = zeros(matrix_size(1),matrix_size(2));

%% calculate median value for each
RateImage_Sig_med = init_mat;
RateImage_BG_med = init_mat;
for xi =1:matrix_size(1)
    for yi=1:matrix_size(2)
        if ~isempty(RateImage_Sig{xi,yi})
            RateImage_Sig_med(xi,yi) = median(abs(RateImage_Sig{xi,yi}));
        end
        if ~isempty(RateImage_BG{xi,yi})
            RateImage_BG_med(xi,yi) = median(abs(RateImage_BG{xi,yi}));
        end
    end
end

RSNR = max(RateImage_Sig_med(:))/std(RateImage_BG_med(:));