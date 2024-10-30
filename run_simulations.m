pyenv('Version', 'C:\Users\30067913\Anaconda3\envs\dvs_performance_metric\python.exe')
% operator = py.importlib.import_module('operator');

% Test 1
for sim = 1:3%6
    pyrunfile(['run_simulation.py T1_',num2str(sim)])
    disp(['done runing simulation run #',num2str(sim),' from test #1'])
end

% % Test 2
% for sim = 1%6
%     pyrunfile(['run_simulation.py T2_',num2str(sim)])
%     disp(['done runing simulation run #',num2str(sim),' from test #1'])
% end

%%

addpath("PERFORMANCE_METRICS\metrics_calc_functions\")
% addpath("..\event_stream\matlab\")

file_list = dir("OUTPUT\events\simdata_T1_*.mat");

PSF_type = {'sharp','slightblur','blurred'};
% val = 0:2:16;
val = 0:1:8;
% val = 3:3:33;
matrix_size = [640,480];%[1280,720];
BG = "constBG";
for psft = 1
    for vi = 1:length(val)
        % load(['OUTPUT\events\simdata_T1_',BG{1},'_',PSF_type{psft},'_t_velocity_',num2str(val(vi)),'.0.mat']);
        % load(['OUTPUT\events\simdata_T1_',BG{1},'_',PSF_type{psft},'_Jitter_speed_',num2str(val(vi)),'.0.mat']);
        load(['OUTPUT\events\simdata_T11_db_m1_t_velocity_',num2str(val(vi)),'.0.mat']);
        all_events.x = simulation_data{1,end}.all_events(:,1)+1;
        all_events.y = simulation_data{1,end}.all_events(:,2)+1;
        all_events.t = simulation_data{1,end}.all_events(:,4);
        all_events.on = simulation_data{1,end}.all_events(:,3);
        sig_ind = simulation_data{1,end}.all_events(:,5);

        ind_to_remove = all_events.t==0;
        all_events.x = all_events.x(~ind_to_remove);
        all_events.y = all_events.y(~ind_to_remove);
        all_events.t = all_events.t(~ind_to_remove);
        all_events.on = all_events.on(~ind_to_remove);
        sig_ind = sig_ind(~ind_to_remove);

        [all_rate_stack,signal_rate_stack,bg_rate_stack,all_ts_stack] = create_rate_image(all_events,matrix_size,sig_ind);
        all_event_count = cellfun(@length,all_rate_stack);
        Signal_event_count = cellfun(@length,signal_rate_stack);
        Bg_event_count = cellfun(@length,bg_rate_stack);
        
        bg.x = all_events.x(~sig_ind);
        bg.y = all_events.y(~sig_ind);
        bg.t = all_events.t(~sig_ind);
        figure;
        plot3(all_events.x(sig_ind==1),all_events.y(sig_ind==1),all_events.t(sig_ind==1),'r.','MarkerSize',0.2); hold on
        plot3(bg.x(1:10:end),bg.y(1:10:end),bg.t(1:10:end),'g.','MarkerSize',0.01);
        % 

        figure;
        plot(all_ts_stack{214,241},all_rate_stack{214,241}); hold on;
        plot(all_ts_stack{379,84},all_rate_stack{379,84});
        % sig_events.x = simulation_data{1,end}.all_events(sig_ind==1,1);
        % sig_events.y = simulation_data{1,end}.all_events(sig_ind==1,2);
        % sig_events.t = simulation_data{1,end}.all_events(sig_ind==1,4);
        % sig_events.on = simulation_data{1,end}.all_events(sig_ind==1,3);
        % % sig_im = sparse(sig_events.x+1,sig_events.y+1,sig_events.on*2-1,1280,720);
        % % imagesc(sig_im) % make sure motion is sub pixel sampled!!
        %
        % bg_events.x = simulation_data{1,end}.all_events(sig_ind==0,1)+1;
        % bg_events.y = simulation_data{1,end}.all_events(sig_ind==0,2)+1;
        % bg_events.t = simulation_data{1,end}.all_events(sig_ind==0,4);
        % bg_events.on = simulation_data{1,end}.all_events(sig_ind==0,3);
        % % bg_im = sparse(bg_events.x,bg_events.y,bg_events.on*2-1,1280,720);
        % % bg_im(1,1) = 0;
        % % imagesc(bg_im)

        [RSNR(psft,vi), RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(signal_rate_stack,bg_rate_stack,matrix_size);
        disp([PSF_type{psft}, ' V=',num2str(val(vi)), 'RSNR: ', num2str(RSNR(psft,vi))])
        figure;
        subplot(2,2,1)
        imagesc(log(RateImage_Sig_med+1)'); colorbar ;
        title([PSF_type{psft},' SIG - not aligned'])
        subplot(2,2,2)
        imagesc(log(RateImage_BG_med+1)'); colorbar;
        title(['BG. RSNR=', num2str(RSNR(psft,vi))])

        [all_events_aligned,filtered_inds] = align_Events(all_events,simulation_data,matrix_size);
        sig_aligned_ind = sig_ind(filtered_inds);
        [all_rate_aligned_stack,signal_rate_aligned_stack,bg_rate_aligned_stack] = create_rate_image(all_events_aligned,matrix_size,sig_aligned_ind);

        [Al_RSNR(psft,vi), RateImage_Sig_aligned_med, RateImage_BG_aligned_med]= calc_RSNR(signal_rate_aligned_stack,bg_rate_aligned_stack,matrix_size);
        disp(['aligned RSNR: ', num2str(Al_RSNR(psft,vi))])
        subplot(2,2,3)
        imagesc(log(RateImage_Sig_aligned_med+1)'); colorbar ;
        title([' V=',num2str(val(vi)), 'SIG - aligned'])
        subplot(2,2,4)
        imagesc(log(RateImage_BG_aligned_med+1)'); colorbar;
        title(['BG. RSNR=', num2str(Al_RSNR(psft,vi))])

    end
end

figure;
plot(val-max(val)/2,RSNR); hold on
plot(val-max(val)/2,Al_RSNR); legend('not aligned RateSNR','aligned RateSNR');

