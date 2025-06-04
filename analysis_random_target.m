

%%

addpath("PERFORMANCE_METRICS\metrics_calc_functions\")
% addpath("..\event_stream\matlab\")
Data_Root_Dir = '..\..\Data\performance_metric_sim\';
% FullTestName = 'frequency_amplitude_heatmap';
% vec = {'0','2','4','6','8','10','12','14','16','18','20'};
% Tests = struct('name','','folder','','date','','bytes',0,'isdir',0,'datenum',0);
% ci = 1;
% for vi =1:length(vec)
%     if ~isempty(dir(['OUTPUT\',FullTestName,'*_',vec{vi},'_nat']))
%         Tests(ci) = dir(['OUTPUT\',FullTestName,'*_',vec{vi},'_nat']);
%         ci = ci+1;
%     end
% end
FullTestName = 'T1';
Tests = dir([Data_Root_Dir,FullTestName,'*']);
epochs = 5;
config_data_file = dir([Data_Root_Dir,Tests(1).name,'\*as_run.ini']);
[test_data,sanned_param] = readINI([config_data_file.folder,'\',config_data_file.name]);
vec_len = length(test_data.(sanned_param{1}).(sanned_param{2}));

% Inisilized full run metrics maps
BW_SNR_all = cell(length(Tests),1);
RSNR_all =  cell(length(Tests),1);
Al_RSNR_all =  cell(length(Tests),1);
Al_SoS_all =  cell(length(Tests),1);
leg = [];


for ti = 1:length(Tests)
    disp(' ')
    disp(['<< Working on results from ',Tests(ti).name,' >>']);
    config_data_file = dir([Data_Root_Dir,Tests(ti).name,'\*as_run.ini']);
    [test_data,sanned_param] = readINI([config_data_file.folder,'\',config_data_file.name]);

    if strcmp(test_data.InitParams.sensor_model,'Manual')
        matrix_size = [test_data.ManualSensorParams.width ,test_data.ManualSensorParams.height];
    elseif strcmp(test_data.InitParams.sensor_model,'Gen4')
        Gen4_config = readINI('config\Gen4_config.ini');
        matrix_size = [Gen4_config.SensorParams.width,Gen4_config.SensorParams.height];
    else
        error('no matrix size defined');
    end

    if ~isempty(sanned_param)
        vector = test_data.(sanned_param{1}).(sanned_param{2});
        ref = 0;
    else
        vector = 0;
        sanned_param = {'',''};
        ref = 1;
    end
    % check that simulation gave enough result files
    count_files = length(dir([Data_Root_Dir,Tests(ti).name,'\events_and_labels\*.mat']));
    if count_files<(epochs*length(vector))
        warning(['missing result files for test ',Tests(ti).name]);
    end


    % Inisilized metrics data for this test
    BW_SNR = zeros(1,length(vector));
    RSNR =  zeros(1,length(vector));
    Al_RSNR = zeros(1,length(vector));
    Al_SoS = zeros(1,length(vector));

    for vi = 1:length(vector)
        n_ep = 0;
        for ep = 1:epochs
            if ~mod(vector(vi),1)
                add_0 = '.0';
            else
                add_0 = '';
            end

            % load all event simulation results
            if ref
                ev_file_name = [Data_Root_Dir,Tests(ti).name,'\events_and_labels\ev_',test_data.InitParams.sim_name,'_',num2str(ep),'.txt'];
                simdata_file_name = [Data_Root_Dir,Tests(ti).name,'\events_and_labels\simdata_',test_data.InitParams.sim_name,'_',num2str(ep),'.mat'];
            else
                ev_file_name = [Data_Root_Dir,Tests(ti).name,'\events_and_labels\ev_',test_data.InitParams.sim_name,'_',sanned_param{2},'_',num2str(vector(vi)),add_0,'_ep',num2str(ep),'.txt'];
                simdata_file_name = [Data_Root_Dir,Tests(ti).name,'\events_and_labels\simdata_',test_data.InitParams.sim_name,'_',sanned_param{2},'_',num2str(vector(vi)),add_0,'_ep',num2str(ep),'.mat'];
            end
            if isfile(ev_file_name) && isfile(simdata_file_name)
                try
                    event_data = load(ev_file_name);
                    load(simdata_file_name);
                catch
                    warning(['Corrupt simulation result files: ',ev_file_name]);
                    continue
                end
            else
                warning(['Missing simulation result files: ',ev_file_name]);
                continue
            end

            n_ep = n_ep+1;

            dt = simulation_data{2}.t - simulation_data{1}.t;
            T = simulation_data{end-1}.t;

            % Read data and convert to event cloud for processing
            all_events.x = event_data(:,1)+1;
            all_events.y = event_data(:,2)+1;
            all_events.on = event_data(:,3);
            all_events.t = event_data(:,4);
            all_events.t = all_events.t - all_events.t(1) + mod(all_events.t(1),100);
            all_events.label = event_data(:,5);
            sig_ind = all_events.label<0;

            ind_to_remove = all_events.t==0; % check if any t=0 values are there - we dont trust these
            if any(find(ind_to_remove))
                all_events.x = all_events.x(~ind_to_remove);
                all_events.y = all_events.y(~ind_to_remove);
                all_events.t = all_events.t(~ind_to_remove);
                all_events.on = all_events.on(~ind_to_remove);
                sig_ind = sig_ind(~ind_to_remove);
            end

            % bg.x = all_events.x(~sig_ind);
            % bg.y = all_events.y(~sig_ind);
            % bg.t = all_events.t(~sig_ind);
            % figure;
            % plot3(all_events.x,all_events.y,all_events.t,'b.','MarkerSize',0.2); hold on
            % plot3(all_events.x(sig_ind==1),all_events.y(sig_ind==1),all_events.t(sig_ind==1),'r.','MarkerSize',5);
            % plot3(bg.x(1:10:end),bg.y(1:10:end),bg.t(1:10:end),'g.','MarkerSize',0.01);


            % get target mask and calculate time each pixel spends on the target
            target_masks = cellfun(@(x)x.binary_target_mask ,simulation_data(1:(end-1)),'UniformOutput' ,false);
            target_time = zeros(size(target_masks{1}));
            sig_ind = zeros(size(sig_ind));
            for k = 1:length(target_masks)
                % inds = find(target_masks{k}==1);
                % inds = randi(matrix_size(1)*matrix_size(2),length(inds),1);
                
                % target_masks{k} = zeros(size(target_masks{k}));
                % target_masks{k}(inds) = 1;

                target_masks{k}(1:(end-49),1:(end-49)) =  target_masks{k}(50:end,50:end);       
                target_time = target_time + double(target_masks{k});

                t0 = simulation_data{k}.t*1e6;
                t1 = simulation_data{k+1}.t*1e6;
                inds = find(all_events.t<t1 & all_events.t>=t0);
                for ii = 1:length(inds)
                    if target_masks{k}(all_events.y(inds(ii)),all_events.x(inds(ii)))
                        sig_ind(inds(ii)) = 1;
                    end
                end


            end
            target_time = target_time'*dt;


            % Create a rate "stack" (event rate vector for each and every pixel
            [all_rate_stack,signal_rate_stack,bg_rate_stack] = create_rate_image(all_events,matrix_size,[],sig_ind);
            all_event_count = cellfun(@length,all_rate_stack(:,:,1));
            Signal_event_count = cellfun(@length,signal_rate_stack(:,:,1));
            Bg_event_count = cellfun(@length,bg_rate_stack(:,:,1));

            % Calculate the Bandwidth SNR metric
            disp([sanned_param{2},' = ', num2str(vector(vi)),', ep',num2str(ep),' results:']);
            
            target_time(target_time<(10*dt)) = 0;

            mean_Signal_event_rate = Signal_event_count./target_time;
            mean_BG_event_rate = Bg_event_count./(T-target_time);
            inds = (~isnan(mean_Signal_event_rate) & ~isinf(mean_Signal_event_rate));
            diffs = (mean_Signal_event_rate(inds)-2*mean_BG_event_rate(inds));
            bw_snr = sum(diffs(diffs>0));
            disp([' - BW_SNR = ',num2str(bw_snr)]);
            BW_SNR(vi) = BW_SNR(vi) + bw_snr; % average value
            % Add some sort of dysplay here for debuging
            
            % mean_BG_event_rate(isnan(mean_BG_event_rate)) = 0;
            % mean_Signal_event_rate(isnan(mean_Signal_event_rate)) = 0;
            % figure(1); imagesc(mean_BG_event_rate);
            % figure(2); imagesc(mean_Signal_event_rate); 

            % % % Calculate Rate SNR metric - TO FIX. Not a good metric...
            % % [rsnr, RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(signal_rate_stack(:,:,1),bg_rate_stack(:,:,1),matrix_size);
            % % disp([' - Rate SNR = ', num2str(rsnr)])
            % % RSNR(vi) = RSNR(vi) + rsnr;
            % figure;
            % subplot(2,2,1)
            % imagesc(log(RateImage_Sig_med+1)'); colorbar ;
            % subplot(2,2,2)
            % imagesc(log(RateImage_BG_med+1)'); colorbar;

            % % % Align events according to target motion in frame
            % % [all_events_aligned,filtered_inds,target_time_al] = align_Events(all_events,simulation_data,matrix_size);
            % % target_time_al(target_time_al<0)=0;
            % % target_time_al = target_time_al'*dt;
            % % sig_aligned_ind = sig_ind(filtered_inds);
            % % [all_rate_aligned_stack,signal_rate_aligned_stack,bg_rate_aligned_stack] = create_rate_image(all_events_aligned,matrix_size,[],sig_aligned_ind);

            % % % sharpness metric
            % % H=full(sparse(all_events_aligned.x(sig_aligned_ind),all_events_aligned.y(sig_aligned_ind),1,matrix_size(1),matrix_size(2)));
            % % Sum_of_squares = sum(H(:).^2.*target_time_al(:)/T);%sum(exp(-H(:)).*target_time_al(:)/T);
            % % disp([' - aligned Sharpness metric = ', num2str(Sum_of_squares)])
            % % Al_SoS(vi) = Al_SoS(vi) + Sum_of_squares;

            % % % Calculate Rate SNR metric for aligned event cloud - TO FIX. Not a good metric...
            % % [al_rsnr, RateImage_Sig_aligned_med, RateImage_BG_aligned_med]= calc_RSNR(signal_rate_aligned_stack(:,:,1),bg_rate_aligned_stack(:,:,1),matrix_size);
            % % disp([' - aligned Rate SNR = ', num2str(al_rsnr)])
            % % Al_RSNR(vi) = Al_RSNR(vi) + al_rsnr;

            % subplot(2,2,3)
            % imagesc(log(RateImage_Sig_aligned_med+1)'); colorbar ;
            % title([' V=',num2str(val(vi)), 'SIG - aligned'])
            % subplot(2,2,4)
            % imagesc(log(RateImage_BG_aligned_med+1)'); colorbar;
            % title(['BG. RSNR=', num2str(Al_RSNR(psft,vi))])

        end
    end

    if length(vector)==1
        BW_SNR_all{ti} = repmat(BW_SNR/n_ep,vec_len,1);
        % % RSNR_all{ti} = repmat(RSNR/n_ep,vec_len,1);
        % % Al_RSNR_all{ti} = repmat(Al_RSNR/n_ep,vec_len,1);
        % % Al_SoS_all{ti} = repmat(Al_SoS/n_ep,vec_len,1);        
    else
        BW_SNR_all{ti} = BW_SNR/n_ep;
        % % RSNR_all{ti} = RSNR/n_ep;
        % % Al_RSNR_all{ti} = Al_RSNR/n_ep;
        % % Al_SoS_all{ti} = Al_SoS/n_ep;
    end

    figure(1)
    loglog(vector,BW_SNR_all{ti}.^-1); hold on; grid on;
    leg = [leg,{replace(Tests(ti).name,'_',' ')}];
    xlabel(replace(sanned_param{2},'_',' '));
    ylabel('\tau_d_e_t_c_t_i_o_n')

    % figure(2)
    % loglog(vector,RSNR_all{ti}); hold on; grid on;
    % xlabel(replace(sanned_param{2},'_',' '));
    % ylabel('Rate SNR')
    % 
    % figure(3)
    % loglog(vector,Al_SoS_all{ti}); hold on; grid on;
    % xlabel(replace(sanned_param{2},'_',' '));
    % ylabel('Aligned Sharpness')
    % drawnow

    disp(['Done evaluating results from ',Tests(ti).name]);
    disp(' ');
    save('temp_results_nat.mat',"BW_SNR_all","RSNR_all","Al_RSNR_all","Al_SoS_all","vector")

end
figure(1)
legend(leg)
figure(2)
legend(leg)
figure(3)
legend(leg)

% BW_SNR_mat(1:13,1) = repmat(BW_SNR_all{1},length(BW_SNR_all{2}),1);
% Al_SoS_mat(1:13,1) = repmat(Al_SoS_all{1},length(BW_SNR_all{2}),1);
% for kk = 1:length(BW_SNR_all)
%     BW_SNR_mat(:,kk) = BW_SNR_all{kk};
%     Al_SoS_mat(:,kk) = Al_SoS_all{kk};
% end
% amp_vec = cellfun(@(x)str2num(x),vector);
% freq_vec = vector;
% figure; imagesc(amp_vec,freq_vec,BW_SNR_mat)
% set(gca,'YScale','log')
% xlabel('Vibration amplitude [pixels]')
% ylabel('Vibration frequency [Hz]')
% title('Sensing frequency [Hz]')
% colormap hot
% colorbar
% 
% figure; imagesc(amp_vec,freq_vec,Al_SoS_mat)
% set(gca,'YScale','log')
% colormap hot
% figure;
% plot(val-max(val)/2,RSNR); hold on
% plot(val-max(val)/2,Al_RSNR); legend('not aligned RateSNR','aligned RateSNR');
%
% figure;
% plot(val-max(val)/2,BW_SNR); %hold on
% plot(val-max(val)/2,Al_BWSNR); legend('not aligned RateSNR','aligned RateSNR');

    figure
    loglog((vector-8+0.001)*50e-3/(300*4.86e-6),BW_SNR_all{1}.^-1); hold on; grid on;
    loglog((vector-8+0.001)*50e-3/(300*4.86e-6),BW_SNR_all{2}.^-1); 
    loglog((vector-8+0.001)*50e-3/(300*4.86e-6),BW_SNR_all{3}.^-1); 
    legend('3 pixels (PSF size)','5 pixels','7 pixels')
    xlabel('pixels/sec');
    ylabel('\tau_d_e_t_c_t_i_o_n [sec]')

   