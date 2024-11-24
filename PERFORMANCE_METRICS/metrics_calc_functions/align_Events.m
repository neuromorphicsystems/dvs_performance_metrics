function [events_aligned,filtered_inds,target_time_al] = align_Events(events,sim_meta_data,matrix_size)
% events            = event data to align
% target_frame_mask = meta data of line-of-sight angle changes
% factor            = LOS angle to pixel shift factor (focal length)/(pixel pitch)

target_masks = cellfun(@(x)x.binary_target_mask ,sim_meta_data(1:(end-1)),'UniformOutput' ,false);


events_aligned_temp.x = zeros(1,length(events.x),'uint16');
events_aligned_temp.y = zeros(1,length(events.y),'uint16');
events_aligned_temp.t = events.t;
events_aligned_temp.on = events.on;

t_new = 0;
offset_x_perv = 0;
offset_y_perv = 0;
ind_end = 0;

for k = 1:(length(sim_meta_data)-1)
    t_old = t_new;
    t_new = sim_meta_data{k}.t*1e6;

    offset_x = sim_meta_data{k}.pixel_offset_x;
    slope_x = (offset_x - offset_x_perv)/(t_new-t_old);
    offset_y = sim_meta_data{k}.pixel_offset_y;
    slope_y = (offset_y - offset_y_perv)/(t_new-t_old);
    
    ind_start = ind_end+1;
    ind_end = find(events.t>=t_new,1)-1;
    events_aligned_temp.x(ind_start:ind_end) = events.x(ind_start:ind_end) + round(offset_x_perv - slope_x*(events.t(ind_start:ind_end)-events.t(ind_start)));
    events_aligned_temp.y(ind_start:ind_end) = events.y(ind_start:ind_end) + round(offset_y_perv - slope_y*(events.t(ind_start:ind_end)-events.t(ind_start)));
    
    ker = zeros(ceil(2*offset_x_perv)+1,ceil(2*offset_y_perv)+1);
    
    if isempty(ker)
        ker = 1;
    else
        midx = ceil(offset_x_perv)+1;
        midy = ceil(offset_y_perv)+1;
        x1 = 
        wx1 =
        x2 = 
        wx2 =
        y1 = 
        wy1 =
        y2 = 
        wy2 =
        ker(x1:x2,y1:y2) = []
        
ker(midx + round(offset_x_perv),midy + round(offset_y_perv))=1;
    end

    target_masks{k} = conv2(target_masks{k},ker,"same");

    offset_x_perv = offset_x;
    offset_y_perv = offset_y;
end

target_time_al = zeros(size(target_masks{1}));
for k = 1:length(target_masks)
    target_time_al = target_time_al + double(target_masks{k});
end
target_time_al = target_time_al'*dt;

filtered_inds = events_aligned_temp.x>0 & events_aligned_temp.x<=matrix_size(1) & events_aligned_temp.y>0 & events_aligned_temp.y<=matrix_size(2);
events_aligned.x = events_aligned_temp.x(filtered_inds);
events_aligned.y = events_aligned_temp.y(filtered_inds);
events_aligned.t = events_aligned_temp.t(filtered_inds);
events_aligned.on = events_aligned_temp.on(filtered_inds);
