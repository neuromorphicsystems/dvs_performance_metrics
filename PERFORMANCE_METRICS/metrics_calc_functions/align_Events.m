function [events_aligned,filtered_inds,target_time_al] = align_Events(events,sim_meta_data,matrix_size)
% events            = event data to align
% target_frame_mask = meta data of line-of-sight angle changes
% factor            = LOS angle to pixel shift factor (focal length)/(pixel pitch)


events_aligned_temp.x = zeros(1,length(events.x),'uint16');
events_aligned_temp.y = zeros(1,length(events.y),'uint16');
events_aligned_temp.t = events.t;
events_aligned_temp.on = events.on;


t_new = 0;
offset_x_perv = 0;
offset_y_perv = 0;
ind_end = 0;

target_masks = cellfun(@(x)x.binary_target_mask ,sim_meta_data,'UniformOutput' ,false);
for k = 2:length(sim_meta_data)
    t_old = t_new;
    t_new = sim_meta_data{k}.t*1e6;
    
    % Caculate the motion base offset and slope (temporal gradient) of the
    % target in the frame compared to the initial frame
    offset_x = sim_meta_data{k}.pixel_offset_x;
    slope_x = (offset_x - offset_x_perv)/(t_new-t_old);
    offset_y = sim_meta_data{k}.pixel_offset_y;
    slope_y = (offset_y - offset_y_perv)/(t_new-t_old);

    % shift all events in the event cloud according to the offset and slope
    ind_start = ind_end+1;
    ind_end = find(events.t>=t_new,1)-1;
    events_aligned_temp.x(ind_start:ind_end) = uint16(events.x(ind_start:ind_end)) + uint16(round(offset_x_perv - slope_x*(events.t(ind_start:ind_end)-events.t(ind_start))));
    events_aligned_temp.y(ind_start:ind_end) = uint16(events.y(ind_start:ind_end)) + uint16(round(offset_y_perv - slope_y*(events.t(ind_start:ind_end)-events.t(ind_start))));

    % align the target mask according to target position
    kernel = create_shift_kernel(offset_y, offset_x);
    target_masks{k} = conv2(target_masks{k},kernel,"same");

    offset_x_perv = offset_x;
    offset_y_perv = offset_y;
end

% sum the target time for each pixel (in units of "frame" - multiply by
% frame time difference to get actual time for each pixel).
target_time_al = zeros(size(target_masks{1}));
for k = 1:length(target_masks)
    target_time_al = target_time_al + double(target_masks{k});
end

%% align the target mask according to target position
filtered_inds = events_aligned_temp.x>0 & events_aligned_temp.x<=matrix_size(1) & events_aligned_temp.y>0 & events_aligned_temp.y<=matrix_size(2);
events_aligned.x = events_aligned_temp.x(filtered_inds);
events_aligned.y = events_aligned_temp.y(filtered_inds);
events_aligned.t = events_aligned_temp.t(filtered_inds);
events_aligned.on = events_aligned_temp.on(filtered_inds);
end

function kernel = create_shift_kernel(y, x)
% Create a kernel for shifting an image by x and y
% x and y can be real values

rows = 2*ceil(abs(y))+1;
cols = 2*ceil(abs(x))+1;

% Generate a meshgrid for the kernel
[X, Y] = meshgrid(1:cols, 1:rows);
Y = Y - ceil(abs(y)) - 1;
X = X - ceil(abs(x)) - 1;
kernel = zeros(size(Y));
if size(kernel,1)>1
    y1 = find(Y(:,1)>=y,1);
    y2 = find(Y(:,1)>=y,1)-1;
else
    y1 = 1; y2 = 1;
end
if size(kernel,2)>1
    x1 = find(X(1,:)>=x,1);
    x2 = find(X(1,:)>=x,1)-1;
else
    x1 = 1; x2 = 1;
end

kernel(y1,x1) = (0.5 + sign(y)*mod(y,1))*(0.5 + sign(x)*mod(x,1));
kernel(y2,x1) = kernel(y2,x1)  + (0.5 - sign(y)*mod(y,1))*(0.5 + sign(x)*mod(x,1));
kernel(y1,x2) = kernel(y1,x2) + (0.5 + sign(y)*mod(y,1))*(0.5 - sign(x)*mod(x,1));
kernel(y2,x2) = kernel(y2,x2)  + (0.5 - sign(y)*mod(y,1))*(0.5 - sign(x)*mod(x,1));
end
