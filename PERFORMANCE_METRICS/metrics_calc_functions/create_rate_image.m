function [all_rate_stack,signal_rate_stack,bg_rate_stack] = create_rate_image(events,matrix_size,signal_indicies)
% This function gets an event cloud and creates a cell structure for the
% entire sensor matrix with a vector of event rates for each pixel.
% With an additional input of labels, additional cell structures for background event rates (labled 0) and signal event rates are provided 
init_mat = zeros(matrix_size(1),matrix_size(2));
temp_last_t = init_mat;
temp_last_p = init_mat;
all_rate_stack = cell(matrix_size(1),matrix_size(2));

if nargin>2
    signal_rate_stack = cell(matrix_size(1),matrix_size(2));
    bg_rate_stack = cell(matrix_size(1),matrix_size(2));
else
    signal_rate_stack = [];
    bg_rate_stack = [];
end

ev_count=0;
for k = 1:length(events.x)
    if temp_last_p(events.x(k),events.y(k))
        ev_count = ev_count+1;
        events_with_rate.r(ev_count) = temp_last_p(events.x(k),events.y(k))*(double(events.t(k)-temp_last_t(events.x(k),events.y(k)))*1e-6)^-1;
        temp_last_t(events.x(k),events.y(k)) = events.t(k);
        temp_last_p(events.x(k),events.y(k)) = events.on(k)*2-1;
        events_with_rate.x(ev_count) = events.x(k);
        events_with_rate.y(ev_count) = events.y(k);
        events_with_rate.t(ev_count) = events.t(k);
        events_with_rate.on(ev_count) = events.on(k);
        all_rate_stack{events.x(k),events.y(k)} = [all_rate_stack{events.x(k),events.y(k)},events_with_rate.r(ev_count)];
    else
        temp_last_t(events.x(k),events.y(k)) = events.t(k);
        temp_last_p(events.x(k),events.y(k)) = events.on(k)*2-1;
        all_rate_stack{events.x(k),events.y(k)} = 0;
    end
    if nargin>2
        if signal_indicies(k)
            signal_rate_stack{events.x(k),events.y(k)} = [signal_rate_stack{events.x(k),events.y(k)},all_rate_stack{events.x(k),events.y(k)}(end)];
        else
            bg_rate_stack{events.x(k),events.y(k)} = [bg_rate_stack{events.x(k),events.y(k)},all_rate_stack{events.x(k),events.y(k)}(end)];
        end
    end
end