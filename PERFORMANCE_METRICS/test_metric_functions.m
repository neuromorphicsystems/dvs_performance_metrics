clear all
close all
addpath('metrics_calc_functions\')

data_path = 'C:\Users\30067913\OneDrive - Western Sydney University\Projects\Perfromance_model\';
file_name = 'event_with_labels';
load('C:\Users\30067913\OneDrive - Western Sydney University\Projects\Perfromance_model\data\masks\target_frame_mask.mat')

%% load events from file.
data_from_file = load([data_path,file_name,'.mat']);
fn = fieldnames(data_from_file);
events = data_from_file.(fn{1});
fn = fieldnames(events);
if contains([fn{:}],'ts')
    time_field = 'ts';
    polarity_field = 'p';
else
    time_field = 't';
    polarity_field = 'on';
end
    
matrix_size = [max([events.x])+1,max([events.y])+1];
T = max([events.(time_field)]);
f = 100e-3;
pitch = 4.86e-6;
factor = f/pitch;

%% seperate event stream to signal and background streams
% also do some adjustments to the field names and data.
label_max = max([events.label]);
events_class = cell(1,label_max+1);
leg = [];
figure;
for l = 1:(label_max+1)
    ind = [events.label]==(l-1);
    events_class{l}.x = [events(ind).x]+1;
    events_class{l}.y = [events(ind).y]+1;
    events_class{l}.t = [events(ind).(time_field)];
    events_class{l}.on = [events(ind).(polarity_field)];
    leg = [leg {['label = ',num2str(l-1)]}];
    plot3(events_class{l}.x,events_class{l}.y,events_class{l}.t,'.','MarkerSize',3); hold on;
end
grid on;
legend(leg)

Sig_events = events_class{2};
BG_events = events_class{1};
disp('done reading event streams')
toc;

%% Take the signal and background event streams and parse them into:
% - Imaged size cell structure with each cell containing the rate vector
%   for the respective pixel.
tic;
RateImage_Sig = create_rate_image(Sig_events,matrix_size);

RateImage_BG = create_rate_image(BG_events,matrix_size);
disp('done parsing event streams')
toc;

%% Rate Signal to Noise Ratio:
% calculate the median rate of the signal pixels, and the median rate of
% the background pixels, and outpu their ratio
tic;
[RSNR, RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(RateImage_Sig,RateImage_BG,matrix_size);
disp(['RSNR: ', num2str(RSNR)])
toc;

%% Aligned Rate Signal to Noise Ratio:
% same as RSNR, but for aligned to motion event stream. Also do:
% - Aligned to motion event stream
% - Alined rate image - same rate image, but after alignment to motion.
tic;
Sig_events_aligned = align_Events(Sig_events,target_frame_mask,matrix_size,factor);
RateImage_Sig_aligned = create_rate_image(Sig_events_aligned,matrix_size);

BG_events_aligned = align_Events(BG_events,target_frame_mask,matrix_size,factor);
RateImage_BG_aligned = create_rate_image(BG_events_aligned,matrix_size);

[RSNR_aligned, RateImage_Sig_med_aligned, RateImage_BG_med_aligned]= calc_RSNR(RateImage_Sig_aligned,RateImage_BG_aligned,matrix_size);
disp(['RSNR_aligned: ', num2str(RSNR_aligned)])
toc;

%% Opposite Pair Signal to Noise Ratio:
% opposite pairs are concecutive events with different polarity. Noise
% events typically tend to have opposite polarities. On the other hand fast
% signals also have alternating polarities between concecutive events. This
% metrics calculates the opposite pairs ratio for signal and background
% (median of each), and outputs the ratio of the two values.
% also do:
% - Image containing the number of consecutive "opposite events" for each
%   pixel, another with the number of consecutive "same event polarity" for
%   each pixel, and another image with the ratio of opposite event of the
%   total number of events for each pixel.
tic;

[N_diffPair_Sig,N_samePair_Sig,opposite_pair_fraction_Sig] = calc_EventPairs(RateImage_Sig);
[N_diffPair_BG,N_samePair_BG,opposite_pair_fraction_BG] = calc_EventPairs(RateImage_Sig);

OPSNR = median_ratio(opposite_pair_fraction_Sig,opposite_pair_fraction_BG);
disp(['OPSNR: ', num2str(OPSNR)])
toc;

% this metric is better for real cameras, and not indicative when tested on
% simulated data.

%% Flickering Signal to Noise Ratio
% we can also look at the total amount of positive events and negative
% events for each pixel and compare their ratio. This doesn't care about
% the ordering of the events, just the toal amount of each.
tic;
FSNR = calc_FlickSNR(Sig_events,BG_events,matrix_size);
disp(['FSNR: ', num2str(FSNR)])
toc;

%% Aligned Signal to Background Sharpness Ratio
% Each event stream, after alignment to motion, is measured for
% "sharpness" - meaning the support of the event stream projected image.
% Good tracking and low vibrations will lead to sharper countur images of a
% target and increase the value of this metrics.
tic;
ASBGS = calc_SharpnessRatio(RateImage_Sig_med_aligned,RateImage_BG_med_aligned);
disp(['ASBGS: ', num2str(ASBGS)])
toc;
