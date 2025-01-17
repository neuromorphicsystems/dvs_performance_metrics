addpath("PERFORMANCE_METRICS\metrics_calc_functions\")
% addpath("..\event_stream\matlab\")

vel = {'10','50','100','500'};
PSF = {'3','6'};
vib = {'0','20'};
epochs = 5;

for veli = 1:length(vel)

    FullTestName = ['MTF_vel',vel{veli},'_PSF3_vib0'];
    disp(' ')
    disp(['<< Working on results from ',FullTestName,' >>']);

    config_data_file = dir(['OUTPUT\',FullTestName,'\*as_run.ini']);
    [test_data,sanned_param] = readINI([config_data_file.folder,'\',config_data_file.name]);
    Testfiles_ev = dir(['OUTPUT\',FullTestName,'\events_and_labels\*.txt']);
    Testfiles_mat = dir(['OUTPUT\',FullTestName,'\events_and_labels\*.mat']);

    vec_len = length(test_data.(sanned_param{1}).(sanned_param{2}));

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
    count_files1 = length(Testfiles_ev);
    count_files2 = length(Testfiles_mat);
    if (count_files1<(epochs*length(vector)) || count_files2<(epochs*length(vector)))
        warning(['missing result files for test ',FullTestName]);
    end


    % Inisilized metrics data for this test
    % BW_SNR = zeros(1,length(vector));
    % RSNR =  zeros(1,length(vector));
    % Al_RSNR = zeros(1,length(vector));
    % Al_SoS = zeros(1,length(vector));

    for vi = 1:5%length(vector)
        n_ep = 0;
        
        All_row_ON_mean = zeros(matrix_size(1),1);
        All_row_OFF_mean = zeros(matrix_size(1),1);
            
        for ep = 1:epochs
            if ~mod(vector(vi),1)
                add_0 = '.0';
            else
                add_0 = '';
            end

            % load all event simulation results
            if ref
                %ev_file_name = ['OUTPUT\',FullTestName,'\events_and_labels\ev_',test_data.InitParams.sim_name,'_',num2str(ep),'.txt'];
                %simdata_file_name = ['OUTPUT\',FullTestName,'\events_and_labels\simdata_',test_data.InitParams.sim_name,'_',num2str(ep),'.mat'];
            else
                ev_file_name = ['OUTPUT\',FullTestName,'\events_and_labels\ev_',test_data.InitParams.sim_name,'_',sanned_param{2},'_',num2str(vector(vi)),add_0,'_ep',num2str(ep),'.txt'];
                simdata_file_name = ['OUTPUT\',FullTestName,'\events_and_labels\simdata_',test_data.InitParams.sim_name,'_',sanned_param{2},'_',num2str(vector(vi)),add_0,'_ep',num2str(ep),'.mat'];
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

            bg.x = all_events.x(~sig_ind);
            bg.y = all_events.y(~sig_ind);
            bg.t = all_events.t(~sig_ind);
            % figure;
            % plot3(all_events.x(sig_ind==1),all_events.y(sig_ind==1),all_events.t(sig_ind==1),'r.','MarkerSize',0.2); hold on
            % plot3(bg.x(1:10:end),bg.y(1:10:end),bg.t(1:10:end),'g.','MarkerSize',0.01);

            % Align events according to target motion in frame
            [all_events_aligned,filtered_inds,target_time_al] = align_Events(all_events,simulation_data,matrix_size);
            target_time_al(target_time_al<0)=0;
            target_time_al = target_time_al'*dt;
            sig_aligned_ind = sig_ind(filtered_inds);
            [all_rate_aligned_stack,signal_rate_aligned_stack,bg_rate_aligned_stack] = create_rate_image(all_events_aligned,matrix_size,[],sig_aligned_ind);           
            
            for yi = 1:matrix_size(2)
                row_ON_mean = cellfun(@(x)mean(x(x>0)),bg_rate_aligned_stack(:,yi,1));
                row_ON_mean(isnan(row_ON_mean)) = 0;
                All_row_ON_mean = All_row_ON_mean + row_ON_mean/matrix_size(2)/epochs;
                row_OFF_mean = cellfun(@(x)mean(x(x<0)),bg_rate_aligned_stack(:,yi,1));
                row_OFF_mean(isnan(row_OFF_mean)) = 0;
                All_row_OFF_mean = All_row_OFF_mean + row_OFF_mean/matrix_size(2)/epochs;
            end

            % Fit to sine wave (know freq, find amp and phase) ?
            % we should do it seperatly for ON and OFF pixels
            
            % for freq higher than pixel count, and given enough events, we
            % can increase the resolution of the test by performing
            % advanced alignment with additional padding of pixels between
            % the pixels.

            

        end
        figure(vi);
        plot(All_row_ON_mean);hold on; 
        plot(All_row_OFF_mean);
        drawnow;
    end
    % 
    % if length(vector)==1
    %     BW_SNR_all{ti} = repmat(BW_SNR/n_ep,vec_len,1);
    %     RSNR_all{ti} = repmat(RSNR/n_ep,vec_len,1);
    %     Al_RSNR_all{ti} = repmat(Al_RSNR/n_ep,vec_len,1);
    %     Al_SoS_all{ti} = repmat(Al_SoS/n_ep,vec_len,1);        
    % else
    %     BW_SNR_all{ti} = BW_SNR/n_ep;
    %     RSNR_all{ti} = RSNR/n_ep;
    %     Al_RSNR_all{ti} = Al_RSNR/n_ep;
    %     Al_SoS_all{ti} = Al_SoS/n_ep;
    % end
    % 
    % figure(1)
    % loglog(vector,BW_SNR_all{ti}); hold on; grid on;
    % leg = [leg,{replace(Test(ti).name,'_',' ')}];
    % xlabel(replace(sanned_param{2},'_',' '));
    % ylabel('BW_S_N_R [Hz]')
    % 
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
    % 
    % disp(['Done evaluating results from ',Test(ti).name]);
    % disp(' ');
    % save('temp_results_nat.mat',"BW_SNR_all","RSNR_all","Al_RSNR_all","Al_SoS_all")

end
% figure(1)
% legend(leg)
% figure(2)
% legend(leg)
% figure(3)
% legend(leg)

% BW_SNR_mat(1:13,1) = repmat(BW_SNR_all{1},length(BW_SNR_all{2}),1);
% Al_SoS_mat(1:13,1) = repmat(Al_SoS_all{1},length(BW_SNR_all{2}),1);
for kk = 1:length(BW_SNR_all)
    BW_SNR_mat(:,kk) = BW_SNR_all{kk};
    Al_SoS_mat(:,kk) = Al_SoS_all{kk};
end
amp_vec = cellfun(@(x)str2num(x),vec);
freq_vec = vector;
figure; imagesc(amp_vec,freq_vec,BW_SNR_mat)
set(gca,'YScale','log')
xlabel('Vibration amplitude [pixels]')
ylabel('Vibration frequency [Hz]')
title('Sensing frequency [Hz]')
colormap hot
colorbar

figure; imagesc(amp_vec,freq_vec,Al_SoS_mat)
set(gca,'YScale','log')
colormap hot
% figure;
% plot(val-max(val)/2,RSNR); hold on
% plot(val-max(val)/2,Al_RSNR); legend('not aligned RateSNR','aligned RateSNR');
%
% figure;
% plot(val-max(val)/2,BW_SNR); %hold on
% plot(val-max(val)/2,Al_BWSNR); legend('not aligned RateSNR','aligned RateSNR');

