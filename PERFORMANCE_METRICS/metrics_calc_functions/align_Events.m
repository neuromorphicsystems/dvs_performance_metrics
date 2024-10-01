function events_aligned = align_Events(events,target_frame_mask,matrix_size,factor)
% events            = event data to align
% target_frame_mask = meta data of line-of-sight angle changes
% factor            = LOS angle to pixel shift factor (focal length)/(pixel pitch)

events_aligned_temp.x = zeros(1,length(events.x),'uint16');
events_aligned_temp.y = zeros(1,length(events.y),'uint16');
events_aligned_temp.t = events.t;
events_aligned_temp.on = events.on;

t_new = target_frame_mask{1}.t*1e6;
ind_end = find(events.t>t_new,1);
for k = 2:length(target_frame_mask)
    t_old = t_new;
    t_new = target_frame_mask{k}.t*1e6;

    ind_start = ind_end; 
    ind_end = find(events.t>t_new,1);

    i_az0 = target_frame_mask{k-1}.i_azimuth;
    i_az1 = target_frame_mask{k}.i_azimuth;
    t_az0 = target_frame_mask{k-1}.t_azimuth;
    t_az1 = target_frame_mask{k}.t_azimuth;
    d_az = (i_az1-t_az1) - (i_az0-t_az0);

    i_el0 = target_frame_mask{k-1}.i_elevation;
    i_el1 = target_frame_mask{k}.i_elevation;
    t_el0 = target_frame_mask{k-1}.t_elevation;
    t_el1 = target_frame_mask{k}.t_elevation;
    d_el = (i_el1-t_el1) - (i_el0-t_el0);

    events_aligned_temp.x(ind_start:ind_end) = events.x(ind_start:ind_end) - uint16(round(factor*d_az*(events.t(ind_start:ind_end)-events.t(ind_start))));
    events_aligned_temp.y(ind_start:ind_end) = events.y(ind_start:ind_end) - uint16(round(factor*d_el*(events.t(ind_start:ind_end)-events.t(ind_start))));
end

ind_outside = events_aligned_temp.x>0 & events_aligned_temp.x<=matrix_size(1) & events_aligned_temp.y>0 & events_aligned_temp.y<=matrix_size(2);
events_aligned.x = events_aligned_temp.x(ind_outside);
events_aligned.y = events_aligned_temp.y(ind_outside);
events_aligned.t = events_aligned_temp.t(ind_outside);
events_aligned.on = events_aligned_temp.on(ind_outside);
