function [RSNR, RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(RateImage_Sig,RateImage_BG,matrix_size)
% get two event rate images: one for signal, and one for background.
% calculate the Rate-Signal-to-Noise ratio for the data.
% 
% init_mat = nan(matrix_size(1),matrix_size(2));

%% calculate median value for each - not a good metric
RateImage_Sig_med = cellfun(@(x)check_med(abs(x),0),RateImage_Sig);
RateImage_BG_med = cellfun(@(x)check_med(abs(x),1),RateImage_BG);

    function med = check_med(x,th)
        if th & ~isempty(x)
            med = median(x);
        elseif length(x)>th
            med = median(x);
        else
            med = 0;
        end
    end

% RSNR = max(RateImage_Sig_med(:))/mean(RateImage_BG_med(:));


%% Calculate the correlation between the spatial temporal excitation function of the signal and the backgound function
% calculate the total rate power 
Sig_image_intensity = cellfun(@(x)calc_energy(x),RateImage_Sig);
Bg_image_intensity = cellfun(@(x)calc_energy(x),RateImage_BG);
% Bg_image_intensity = cellfun(@(x)(abs(x(1))/2+sum(abs(0.5*(x(1:(end-1)).^2-sign(x(1:(end-1)).*x(2:end)).*x(2:end).^2)./(x(2:end).*(x(1:(end-1))-x(2:end)))))^2),RateImage_BG);

% RSNR = sum(Sig_image_intensity(:),"omitnan")./mean(Bg_image_intensity(:),"omitnan");
temp_energy_diff = (Sig_image_intensity(:) - Bg_image_intensity(:));
RSNR = sum(temp_energy_diff(temp_energy_diff>0));

    function energy = calc_energy(rate_vec)
        if length(rate_vec)>1
            energy = abs(rate_vec(1))/2;
            energy = energy + sum(abs(0.5*(rate_vec(1:(end-1)).^2-sign(rate_vec(1:(end-1)).*rate_vec(2:end)).*rate_vec(2:end).^2)./(rate_vec(2:end).*(rate_vec(1:(end-1))-rate_vec(2:end)))));
            % energy = energy^2;
        else
            energy = 0;
        end
    end
end