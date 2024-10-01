function RateImage = create_rate_image(events,matrix_size)

init_mat = zeros(matrix_size(1),matrix_size(2));
temp_last_t = init_mat;
temp_last_p = init_mat;
RateImage = cell(matrix_size(1),matrix_size(2));
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
        RateImage{events.x(k),events.y(k)} = [RateImage{events.x(k),events.y(k)},events_with_rate.r(ev_count)];
    else
        temp_last_t(events.x(k),events.y(k)) = events.t(k);
        temp_last_p(events.x(k),events.y(k)) = events.on(k)*2-1;
        RateImage{events.x(k),events.y(k)} = 0;
    end
end