function [RSNR, RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(Signal_events,BG_events,matrix_size)
% get two event streams: one for target, and one for background.
% calculate the Rate-Signal-to-Noise ratio for the data.
% 
init_mat = zeros(matrix_size(1),matrix_size(2));


%% calculate rate image for signal
temp_last_t = init_mat;
temp_last_p = init_mat;
RateImage_Sig = cell(matrix_size(1),matrix_size(2));
ev_count=0;
for k = 1:length(Signal_events.x)
    if temp_last_p(Signal_events.x(k),Signal_events.y(k))
        ev_count = ev_count+1;
        events_with_rate.r(ev_count) = temp_last_p(Signal_events.x(k),Signal_events.y(k))*(double(Signal_events.t(k)-temp_last_t(Signal_events.x(k),Signal_events.y(k)))*1e-6)^-1;
        temp_last_t(Signal_events.x(k),Signal_events.y(k)) = Signal_events.t(k);
        temp_last_p(Signal_events.x(k),Signal_events.y(k)) = Signal_events.on(k)*2-1;
        events_with_rate.x(ev_count) = Signal_events.x(k);
        events_with_rate.y(ev_count) = Signal_events.y(k);
        events_with_rate.t(ev_count) = Signal_events.t(k);
        events_with_rate.on(ev_count) = Signal_events.on(k);
        RateImage_Sig{Signal_events.x(k),Signal_events.y(k)} = [RateImage_Sig{Signal_events.x(k),Signal_events.y(k)},events_with_rate.r(ev_count)];
    else
        temp_last_t(Signal_events.x(k),Signal_events.y(k)) = Signal_events.t(k);
        temp_last_p(Signal_events.x(k),Signal_events.y(k)) = Signal_events.on(k)*2-1;
        RateImage_Sig{Signal_events.x(k),Signal_events.y(k)} = 0;
    end
end

%% calculate rate image for background
temp_last_t = init_mat;
temp_last_p = init_mat;
RateImage_BG = cell(matrix_size(1),matrix_size(2));
ev_count=0;
for k = 1:length(BG_events.x)
    if temp_last_p(BG_events.x(k),BG_events.y(k))
        ev_count = ev_count+1;
        events_with_rate.r(ev_count) = temp_last_p(BG_events.x(k),BG_events.y(k))*(double(BG_events.t(k)-temp_last_t(BG_events.x(k),BG_events.y(k)))*1e-6)^-1;
        temp_last_t(BG_events.x(k),BG_events.y(k)) = BG_events.t(k);
        temp_last_p(BG_events.x(k),BG_events.y(k)) = BG_events.on(k)*2-1;
        events_with_rate.x(ev_count) = BG_events.x(k);
        events_with_rate.y(ev_count) = BG_events.y(k);
        events_with_rate.t(ev_count) = BG_events.t(k);
        events_with_rate.on(ev_count) = BG_events.on(k);
        RateImage_BG{BG_events.x(k),BG_events.y(k)} = [RateImage_BG{BG_events.x(k),BG_events.y(k)},events_with_rate.r(ev_count)];
    else
        temp_last_t(BG_events.x(k),BG_events.y(k)) = BG_events.t(k);
        temp_last_p(BG_events.x(k),BG_events.y(k)) = BG_events.on(k)*2-1;
        RateImage_BG{BG_events.x(k),BG_events.y(k)} = 0;
    end
end


%% calculate median value for each
RateImage_Sig_med = init_mat;
RateImage_BG_med = init_mat;
for xi =1:matrix_size(1)
    for yi=1:matrix_size(2)
        if length(RateImage_Sig{xi,yi})>2
            RateImage_Sig_med(xi,yi) = median(abs(RateImage_Sig{xi,yi}));
        end
        if ~isempty(RateImage_BG{xi,yi})
            RateImage_BG_med(xi,yi) = median(abs(RateImage_BG{xi,yi}));
        end
    end
end

RSNR = max(RateImage_Sig_med(:))/std(RateImage_BG_med(:));