
file_to_load = 'T1_constBG_blurred_t_velocity_12.0_ep2';
Dirdir = 'T1_3';
% file_to_load = 'T1_constBG_sharp_t_velocity_10.0_ep1';
% Dirdir = 'T1_1';

Data_Root_Dir = 'C:\Users\30067913\Data\performance_metric_sim\';
load([Data_Root_Dir,Dirdir,'\events_and_labels\simdata_',file_to_load,'.mat']);
eventdata = load([Data_Root_Dir,Dirdir,'\events_and_labels\ev_',file_to_load,'.txt']);
matrix_size = [640,480];

all_events.x = eventdata(:,1)+1;
all_events.y = eventdata(:,2)+1;
all_events.on = eventdata(:,3);
all_events.t = eventdata(:,4);
all_events.t = all_events.t - all_events.t(1) + mod(all_events.t(1),100);
all_events.label = eventdata(:,5);
sig_ind = all_events.label<0;

ind_to_remove = all_events.t==0; % check if any t=0 values are there - we dont trust these
if any(find(ind_to_remove))
    all_events.x = all_events.x(~ind_to_remove);
    all_events.y = all_events.y(~ind_to_remove);
    all_events.t = all_events.t(~ind_to_remove);
    all_events.on = all_events.on(~ind_to_remove);
    sig_ind = sig_ind(~ind_to_remove);
end
[all_events_aligned,filtered_inds,target_time_al] = align_Events(all_events,simulation_data,matrix_size);
sig_aligned_ind = sig_ind(filtered_inds);

%%
t0 = 300e-3;
% dT = [5,10,15,20,25,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]*1e-3;
dT = [0.25:0.25:150]*1e-3;
% dT = [5,25,50,100,200,300,400,500,600,700,800,900,1000,1250,1500,2000,2250]*1e-3;
do_vid= 1;

if do_vid
    vid = VideoWriter("accum2.avi");
    vid.FrameRate = 15;
    open(vid)
end

for ki = 1:length(dT)
    inds = find(all_events_aligned.t>=(t0*1e6) & all_events_aligned.t<((t0+dT(ki))*1e6));

    % events.x = all_events_aligned.x(inds);
    % events.t = all_events_aligned.t(inds);
    % events.y = all_events_aligned.y(inds);
    %
    % rate_energy = zeros(matrix_size(1),matrix_size(2));
    % for xi = 1:matrix_size(1)
    %     for yi = 1:matrix_size(2)
    %         inds = find(events.x==xi & events.y==yi);
    %         t = unique(events.t(inds));
    %         if length(t)>1
    %             rate_energy(xi,yi) = sum(((t(2:end)-t(1:(end-1)))*1e-6).^-2);
    %         end
    %     end
    % end
    % figure;
    % imagesc(log(rate_energy))


    H=full(sparse(all_events_aligned.x(inds),all_events_aligned.y(inds),1,matrix_size(1),matrix_size(2)));
    % figure;
    % plot3(all_events_aligned.x,all_events_aligned.y,all_events_aligned.t,'b.','MarkerSize',0.3); hold on;
    % plot3(all_events_aligned.x(sig_aligned_ind),all_events_aligned.y(sig_aligned_ind),all_events_aligned.t(sig_aligned_ind),'r.','MarkerSize',0.3);
    %  figure;
    % plot3(all_events.x,all_events.y,all_events.t,'b.','MarkerSize',0.3); hold on;
    % plot3(all_events.x(sig_ind),all_events.y(sig_ind),all_events.t(sig_ind),'r.','MarkerSize',3);
    % ylim([238 244])
    % xlim([400 450])
    % zlim([0.7 0.9]*1e6)
    figure(1);
    imagesc(H',[0 4]); colormap gray;
    xlim([295 351])
    ylim([209 273])
    axis off
    title([num2str(round(dT(ki)*1e3)),' ms'])
    drawnow;
    % pause(1)
    if do_vid
        frame = getframe(gcf);
        writeVideo(vid,frame)
    end
  
end
if do_vid
    close(vid)
end

%% spiking plot

xx = 330:4:350;
yy = 239:242;
dt = 2.5e3;

target_mask = zeros(length(simulation_data),length(xx)*length(yy));
for k = 1:length(simulation_data)
    tm = simulation_data{k}.binary_target_mask(yy,xx);
    target_mask(k,:) =tm(:);
end
figure;
imagesc(target_mask)

t_spikes_all = zeros(length(xx)*length(yy),2502500);
t_spikes_tar = nan(length(xx)*length(yy),2502500);

figure;
% tiledlayout("vertical")
for xi = 1:(length(xx)*length(yy))
    x1 = xx(floor((xi-1)/length(yy)+1));
    y1 = yy(mod(xi-1,length(yy))+1);
    inds = all_events.x==x1 & all_events.y==y1;
    times = all_events.t(inds);
    t_spikes_all(xi,times) = 1;
    is_tar = repelem(target_mask(2:end,xi),2500)==1;
    t_spikes_tar(xi,is_tar) = t_spikes_all(xi,is_tar);

    ax(xi) = subplot(length(xx)*length(yy),1,xi);
     % = nexttile;
    plot(t_spikes_all(xi,:),'k'); hold on; %,'LineWidth',1.5); hold on;
    plot(find(t_spikes_all(xi,:)>0),t_spikes_all(xi,t_spikes_all(xi,:)>0),'^k');
    plot(t_spikes_tar(xi,:),'r','LineWidth',2)
    plot(find(t_spikes_tar(xi,:)>0),t_spikes_tar(xi,t_spikes_tar(xi,:)>0),'^r');
    axis off
end
linkaxes(ax)
