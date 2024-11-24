% pyenv('Version', 'C:\Users\30067913\Anaconda3\envs\dvs_performance_metric\python.exe')
% % operator = py.importlib.import_module('operator');
% 
% % Test 1
% for sim = 1:3%6
%     pyrunfile(['run_simulation.py T1_',num2str(sim)])
%     disp(['done runing simulation run #',num2str(sim),' from test #1'])
% end
% 
% % % Test 2
% % for sim = 1%6
% %     pyrunfile(['run_simulation.py T2_',num2str(sim)])
% %     disp(['done runing simulation run #',num2str(sim),' from test #1'])
% % end

%%

addpath("PERFORMANCE_METRICS\metrics_calc_functions\")
% addpath("..\event_stream\matlab\")

file_list = dir("OUTPUT\T1_1\simdata_T1_*.mat");

PSF_type = {'sharp','slightblur','blurred'};
val = 0:2:16;

% val = 0:1:8;
% val = 3:3:33;
matrix_size = [1280,720]; %[640,480];
BG = "constBG";
for psft = 1
    for vi = 1:length(val)
        load(['OUTPUT\events\simdata_T1_',BG{1},'_',PSF_type{psft},'_t_velocity_',num2str(val(vi)),'.0.mat']);
        % load(['OUTPUT\events\simdata_T1_',BG{1},'_',PSF_type{psft},'_Jitter_speed_',num2str(val(vi)),'.0.mat']);
        % load(['OUTPUT\T11_1\simdata_T11_db_m1_t_velocity_',num2str(val(vi)),'.0.mat']);
        dt = simulation_data{2}.t - simulation_data{1}.t;
        T = simulation_data{end-1}.t;

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

        % get the target mask and see how much time each pixel sees the
        % target
        target_masks = cellfun(@(x)x.binary_target_mask ,simulation_data(1:(end-1)),'UniformOutput' ,false);
        target_time = zeros(size(target_masks{1}));
        for k = 1:length(target_masks)
            target_time = target_time + double(target_masks{k});
        end
        target_time = target_time'*dt;

        [all_rate_stack,signal_rate_stack,bg_rate_stack] = create_rate_image(all_events,matrix_size,[],sig_ind);
        all_event_count = cellfun(@length,all_rate_stack(:,:,1));
        Signal_event_count = cellfun(@length,signal_rate_stack(:,:,1));
        Bg_event_count = cellfun(@length,bg_rate_stack(:,:,1));
        
        mean_Signal_event_rate = Signal_event_count./target_time;
        mean_BG_event_rate = Bg_event_count./(T-target_time);
        inds = ~isnan(mean_Signal_event_rate);
        BWSNR(psft,vi) = sum(abs(mean_Signal_event_rate(inds)-mean_BG_event_rate(inds)));


        bg.x = all_events.x(~sig_ind);
        bg.y = all_events.y(~sig_ind);
        bg.t = all_events.t(~sig_ind);
        figure;
        plot3(all_events.x(sig_ind==1),all_events.y(sig_ind==1),all_events.t(sig_ind==1),'r.','MarkerSize',0.2); hold on
        plot3(bg.x(1:10:end),bg.y(1:10:end),bg.t(1:10:end),'g.','MarkerSize',0.01);
        % 

        % lets add a new "SNR" metric - bandwidth SNR. Stating the time it
        % would take to detect the target, or at what frequancy can I
        % compute so to see the target.
        % we define it as sum_pixels(integrate_signal_rates/T_signal - 
        % integrate_background_rates/T_bg)
        % later make sure to put this into its own function <<<<<
        % 
        % signal_pix = cellfun(@(x)~isempty(x),signal_rate_stack(:,:,1));
        % temp_stack = reshape(signal_rate_stack, matrix_size(1)*matrix_size(2),2);
        % signal_rates = temp_stack(signal_pix(:),1);
        % signal_ts = temp_stack(signal_pix(:),2);
        % 
        % dT_sig = cellfun(@(x)max(x),signal_ts) - cellfun(@(x)min(x),signal_ts) + cellfun(@(x)x(1)^-1,signal_rates);
        % N_event_sig = cellfun(@(x)length(x),signal_ts);
        % event_av_sig = N_event_sig./dT_sig;
        % 
        % 
        % bg_pix = cellfun(@(x)~isempty(x),bg_rate_stack(:,:,1));
        % temp_stack = reshape(bg_rate_stack, matrix_size(1)*matrix_size(2),2);
        % bg_rates = temp_stack(signal_pix(:),1);
        % bg_ts = temp_stack(signal_pix(:),2);
        % 
        % dT_bg = T;
        % N_event_bg = cellfun(@(x)length(x),bg_ts);
        % event_av_bg = N_event_bg./dT_bg;
        % 
        % BWSNR(psft,vi) = sum(event_av_sig-event_av_bg);

        

        % figure;
        % plot(all_ts_stack{214,241},all_rate_stack{214,241}); hold on;
        % plot(all_ts_stack{379,84},all_rate_stack{379,84});
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

        [RSNR(psft,vi), RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(signal_rate_stack(:,:,1),bg_rate_stack(:,:,1),matrix_size);
        disp([PSF_type{psft}, ' V=',num2str(val(vi)), ', RSNR: ', num2str(RSNR(psft,vi))])
        figure;
        subplot(2,2,1)
        imagesc(log(RateImage_Sig_med+1)'); colorbar ;
        title([PSF_type{psft},' SIG - not aligned'])
        subplot(2,2,2)
        imagesc(log(RateImage_BG_med+1)'); colorbar;
        title(['BG. RSNR=', num2str(RSNR(psft,vi))])

        [all_events_aligned,filtered_inds,target_time_al] = align_Events(all_events,simulation_data,matrix_size);
        sig_aligned_ind = sig_ind(filtered_inds);
        [all_rate_aligned_stack,signal_rate_aligned_stack,bg_rate_aligned_stack] = create_rate_image(all_events_aligned,matrix_size,[],sig_aligned_ind);
        
        Signal_event_count_al = cellfun(@length,signal_rate_aligned_stack(:,:,1));
        Bg_event_count_al = cellfun(@length,bg_rate_aligned_stack(:,:,1));
        mean_Signal_event_rate_al = Signal_event_count_al./target_time_al;
        mean_BG_event_rate_al = Bg_event_count_al./(T-target_time_al);
        inds = ~isnan(mean_Signal_event_rate_al);
        BWSNR_al(psft,vi) = sum(abs(mean_Signal_event_rate_al(inds)-mean_BG_event_rate_al(inds)));

        % signal_pix = cellfun(@(x)~isempty(x),signal_rate_aligned_stack(:,:,1));
        % temp_stack = reshape(signal_rate_aligned_stack, matrix_size(1)*matrix_size(2),2);
        % signal_rates = temp_stack(signal_pix(:),1);
        % signal_ts = temp_stack(signal_pix(:),2);
        % 
        % dT_sig = cellfun(@(x)max(x),signal_ts) - cellfun(@(x)min(x),signal_ts) + cellfun(@(x)x(1)^-1,signal_rates);
        % N_event_sig = cellfun(@(x)length(x),signal_ts);
        % event_av_sig = N_event_sig./dT_sig;
        % 
        % 
        % bg_pix = cellfun(@(x)~isempty(x),bg_rate_aligned_stack(:,:,1));
        % temp_stack = reshape(bg_rate_aligned_stack, matrix_size(1)*matrix_size(2),2);
        % bg_rates = temp_stack(signal_pix(:),1);
        % bg_ts = temp_stack(signal_pix(:),2);
        % 
        % dT_bg = T;
        % N_event_bg = cellfun(@(x)length(x),bg_ts);
        % event_av_bg = N_event_bg./dT_bg;
        % 
        % Al_BWSNR(psft,vi) = sum(event_av_sig-event_av_bg);






        [Al_RSNR(psft,vi), RateImage_Sig_aligned_med, RateImage_BG_aligned_med]= calc_RSNR(signal_rate_aligned_stack(:,:,1),bg_rate_aligned_stack(:,:,1),matrix_size);
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

figure;
plot(val-max(val)/2,BWSNR); hold on
plot(val-max(val)/2,Al_BWSNR); legend('not aligned RateSNR','aligned RateSNR');

