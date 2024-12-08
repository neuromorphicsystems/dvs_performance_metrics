<<<<<<< HEAD
function RateImage = create_rate_image(events,matrix_size)

init_mat = zeros(matrix_size(1),matrix_size(2));
temp_last_t = init_mat;
temp_last_p = init_mat;
RateImage = cell(matrix_size(1),matrix_size(2));
ev_count=0;
for k = 1:length(events.x)
    if temp_last_p(events.x(k),events.y(k))
=======
function [all_rates_stack,signal_rate_stack,bg_rate_stack] = create_rate_image(events,matrix_size,all_rates_stack,signal_indicies)
% This function gets an event cloud and creates a cell structure for the
% entire sensor matrix with a vector of event rates for each pixel.
% With an additional input of labels, additional cell structures for background event rates (labled 0) and signal event rates are provided 
init_mat = zeros(matrix_size(1),matrix_size(2));
temp_last_t = init_mat;
temp_last_p = init_mat;
% temp_last_count = init_mat;
if nargin>2 && ~isempty(all_rates_stack)
    % use input rate stack
    all_rates = all_rates_stack(:,:,1);
    all_ts = all_rates_stack(:,:,2);
    t_start = max([all_ts{:}])+10;
    if isempty(t_start)
        t_start = 0;
    end
else
    all_rates = cell(matrix_size(1),matrix_size(2));
    all_ts = cell(matrix_size(1),matrix_size(2));
    t_start = 0;
end
t_int = double(events.t(1))*1e-6;

if nargin>3
    signal_rate_stack = cell(matrix_size(1),matrix_size(2),2);
    bg_rate_stack = cell(matrix_size(1),matrix_size(2),2);
else
    signal_rate_stack = [];
    sig_ts = [];
    bg_rate_stack = [];
    bg_ts = [];
end

ev_count=0;
for k = 1:length(events.x)
    if temp_last_p(events.x(k),events.y(k)) && temp_last_t(events.x(k),events.y(k))<events.t(k)
>>>>>>> main
        ev_count = ev_count+1;
        events_with_rate.r(ev_count) = temp_last_p(events.x(k),events.y(k))*(double(events.t(k)-temp_last_t(events.x(k),events.y(k)))*1e-6)^-1;
        temp_last_t(events.x(k),events.y(k)) = events.t(k);
        temp_last_p(events.x(k),events.y(k)) = events.on(k)*2-1;
        events_with_rate.x(ev_count) = events.x(k);
        events_with_rate.y(ev_count) = events.y(k);
        events_with_rate.t(ev_count) = events.t(k);
        events_with_rate.on(ev_count) = events.on(k);
<<<<<<< HEAD
        RateImage{events.x(k),events.y(k)} = [RateImage{events.x(k),events.y(k)},events_with_rate.r(ev_count)];
    else
        temp_last_t(events.x(k),events.y(k)) = events.t(k);
        temp_last_p(events.x(k),events.y(k)) = events.on(k)*2-1;
        RateImage{events.x(k),events.y(k)} = 0;
    end
end
=======
        all_rates{events.x(k),events.y(k)} = [all_rates{events.x(k),events.y(k)},events_with_rate.r(ev_count)];
        all_ts{events.x(k),events.y(k)} = [all_ts{events.x(k),events.y(k)},double(events.t(k))*1e-6];
        if nargin>3
            if signal_indicies(k)
                signal_rate_stack{events.x(k),events.y(k),1} = [signal_rate_stack{events.x(k),events.y(k),1},all_rates{events.x(k),events.y(k)}(end)];
                signal_rate_stack{events.x(k),events.y(k),2} = [signal_rate_stack{events.x(k),events.y(k),2},all_ts{events.x(k),events.y(k)}(end)];
            else
                bg_rate_stack{events.x(k),events.y(k),1} = [bg_rate_stack{events.x(k),events.y(k),1},all_rates{events.x(k),events.y(k)}(end)];
                bg_rate_stack{events.x(k),events.y(k),2} = [bg_rate_stack{events.x(k),events.y(k),2},all_ts{events.x(k),events.y(k)}(end)];
            end
        end
    else
        temp_last_t(events.x(k),events.y(k)) = events.t(k);
        temp_last_p(events.x(k),events.y(k)) = events.on(k)*2-1;
    end


end
if ~isempty(all_rates_stack)
    all_rates_stack(:,:,1) = all_rates;
else
    all_rates_stack = all_rates;
end
all_rates_stack(:,:,2) = cellfun(@(x)(x-t_int+t_start), all_ts, 'UniformOutput', false);
    
>>>>>>> main
