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
% val = 0:2:16;
val = [0,4,6,7,7.5,7.8,8,8.2,8.5,9,10,12,16];
% val = 0:1:8;
% val = 3:3:33;
matrix_size = [1280,720]; %[640,480];
BG = "constBG";


% Inisilized metrics data
BW_SNR = zeros(length(PSF_type),length(va));
RSNR =  zeros(length(PSF_type),length(va));
Al_RSNR = zeros(length(PSF_type),length(va));

% loop on simulation results
for psft = 1:3
    for vi = 1:length(val)

        % read simulation data file
        if mod(val(vi),1)
            do_point ='';
        else
            do_point = '.0';
        end
        load(['OUTPUT\T1_',num2str(psft),'\simdata_T1_',BG{1},'_',PSF_type{psft},'_t_velocity_',num2str(val(vi)),do_point,'.mat']);
        dt = simulation_data{2}.t - simulation_data{1}.t;
        T = simulation_data{end-1}.t;

        % Read data and convert to event cloud for processing
        x = cellfun(@(celobj)celobj.x+1,simulation_data(2:end),'UniformOutput',false);
        all_events.x = [x{:}];
        y = cellfun(@(celobj)celobj.y+1,simulation_data(2:end),'UniformOutput',false);
        all_events.y = [y{:}];
        ts = cellfun(@(celobj)celobj.ts,simulation_data(2:end),'UniformOutput',false);
        all_events.t = [ts{:}];
        all_events.t = all_events.t - all_events.t(1) + mod(all_events.t(1),100);
        on = cellfun(@(celobj)celobj.p,simulation_data(2:end),'UniformOutput',false);
        all_events.on = [on{:}];
        label = cellfun(@(celobj)celobj.l,simulation_data(2:end),'UniformOutput',false);
        all_events.label = [label{:}];
        % all_events.x = simulation_data{1,end}.all_events(:,1)+1;
        % all_events.y = simulation_data{1,end}.all_events(:,2)+1;
        % all_events.t = simulation_data{1,end}.all_events(:,4);
        % all_events.on = simulation_data{1,end}.all_events(:,3);
        % load labels as well
        sig_ind = all_events.label~=0;
        
        ind_to_remove = all_events.t==0; % check if any t=0 values are there - we dont trust these
        all_events.x = all_events.x(~ind_to_remove);
        all_events.y = all_events.y(~ind_to_remove);
        all_events.t = all_events.t(~ind_to_remove);
        all_events.on = all_events.on(~ind_to_remove);
        sig_ind = sig_ind(~ind_to_remove);

        % bg.x = all_events.x(~sig_ind);
        % bg.y = all_events.y(~sig_ind);
        % bg.t = all_events.t(~sig_ind);
        % figure;
        % plot3(all_events.x(sig_ind==1),all_events.y(sig_ind==1),all_events.t(sig_ind==1),'r.','MarkerSize',0.2); hold on
        % plot3(bg.x(1:10:end),bg.y(1:10:end),bg.t(1:10:end),'g.','MarkerSize',0.01);


        % get target mask and calculate time each pixel spends on the target
        target_masks = cellfun(@(x)x.binary_target_mask ,simulation_data(1:(end-1)),'UniformOutput' ,false);
        target_time = zeros(size(target_masks{1}));
        for k = 1:length(target_masks)
            target_time = target_time + double(target_masks{k});
        end
        target_time = target_time'*dt;


        % Create a rate "stack" (event rate vector for each and every pixel       
        [all_rate_stack,signal_rate_stack,bg_rate_stack] = create_rate_image(all_events,matrix_size,[],sig_ind);
        all_event_count = cellfun(@length,all_rate_stack(:,:,1));
        Signal_event_count = cellfun(@length,signal_rate_stack(:,:,1));
        Bg_event_count = cellfun(@length,bg_rate_stack(:,:,1));
        
        % Calculate the Bandwidth SNR metric
        mean_Signal_event_rate = Signal_event_count./target_time;
        mean_BG_event_rate = Bg_event_count./(T-target_time);
        inds = ~isnan(mean_Signal_event_rate);
        BW_SNR(psft,vi) = sum(abs(mean_Signal_event_rate(inds)-mean_BG_event_rate(inds)));
        % Add some sort of dysplay here for debuging

        % Calculate Rate SNR metric
        [RSNR(psft,vi), RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(signal_rate_stack(:,:,1),bg_rate_stack(:,:,1),matrix_size);
        disp([PSF_type{psft}, ' V=',num2str(val(vi)), ', RSNR: ', num2str(RSNR(psft,vi))])
        figure;
        subplot(2,2,1)
        imagesc(log(RateImage_Sig_med+1)'); colorbar ;
        title([PSF_type{psft},' SIG - not aligned'])
        subplot(2,2,2)
        imagesc(log(RateImage_BG_med+1)'); colorbar;
        title(['BG. RSNR=', num2str(RSNR(psft,vi))])

        % Align events according to target motion in frame
        [all_events_aligned,filtered_inds,target_time_al] = align_Events(all_events,simulation_data,matrix_size);
        target_time_al = target_time_al'*dt;
        sig_aligned_ind = sig_ind(filtered_inds);
        [all_rate_aligned_stack,signal_rate_aligned_stack,bg_rate_aligned_stack] = create_rate_image(all_events_aligned,matrix_size,[],sig_aligned_ind);
        
        % Calculate Rate SNR metric for aligned event cloud
        [Al_RSNR(psft,vi), RateImage_Sig_aligned_med, RateImage_BG_aligned_med]= calc_RSNR(signal_rate_aligned_stack(:,:,1),bg_rate_aligned_stack(:,:,1),matrix_size);
        disp(['aligned RSNR: ', num2str(Al_RSNR(psft,vi))])
        subplot(2,2,3)
        imagesc(log(RateImage_Sig_aligned_med+1)'); colorbar ;
        title([' V=',num2str(val(vi)), 'SIG - aligned'])
        subplot(2,2,4)
        imagesc(log(RateImage_BG_aligned_med+1)'); colorbar;
        title(['BG. RSNR=', num2str(Al_RSNR(psft,vi))])
        
        % Signal_event_count_al = cellfun(@length,signal_rate_aligned_stack(:,:,1));
        % Bg_event_count_al = cellfun(@length,bg_rate_aligned_stack(:,:,1));
        % mean_Signal_event_rate_al = Signal_event_count_al./target_time_al;
        % mean_BG_event_rate_al = Bg_event_count_al./(T-target_time_al);
        % inds = ~isnan(mean_Signal_event_rate_al);
        % BWSNR_al(psft,vi) = sum(abs(mean_Signal_event_rate_al(inds)-mean_BG_event_rate_al(inds)));

    end
end

figure;
plot(val-max(val)/2,RSNR); hold on
plot(val-max(val)/2,Al_RSNR); legend('not aligned RateSNR','aligned RateSNR');

figure;
 plot(val-max(val)/2,BW_SNR); %hold on
% plot(val-max(val)/2,Al_BWSNR); legend('not aligned RateSNR','aligned RateSNR');

