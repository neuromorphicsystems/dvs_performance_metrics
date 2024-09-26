clear all
close all
addpath('metrics_calc_functions\')

data_path = 'data\events\';
file_name = 'ev_500.0_350.0_0.6_0.5_0.02';

data_from_file = load([data_path,file_name,'.mat']);
fn = fieldnames(data_from_file);
events = data_from_file.(fn{1});
matrix_size = [max(events.x)+1,max(events.y)+1];
T = max(events.ts);


label_max = max([events.label]);
events_class = cell(1,label_max+1);
leg = [];
figure;
for l = 1:(label_max+1)
    ind = [events.label]==(l-1);
    events_class{l}.x = [events(ind).x]+1;
    events_class{l}.y = [events(ind).y]+1;
    events_class{l}.t = [events(ind).t];
    events_class{l}.on = [events(ind).on];
    leg = [leg {['label = ',num2str(l-1)]}];
    plot3(events_class{l}.x,events_class{l}.y,events_class{l}.t,'.','MarkerSize',3); hold on;
end
grid on;
legend(leg)

Signal_events = events_class{2};
BG_events = events_class{1};

[RSNR, RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(Signal_events,BG_events,matrix_size);
FSNR = calc_FlickSNR(Signal_events,BG_events,matrix_size);
ASBG = calc_Sharpness(RateImage_Sig_med,RateImage_BG_med)