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

%
% file_list = dir("OUTPUT\T1_1\simdata_T1_*.mat");
%
% PSF_type = {'sharp','slightblur','blurred'};
% % val = 0:2:16;
% val = [0,4,6,7,7.5,7.8,8,8.2,8.5,9,10,12,16];
% % val = 0:1:8;
% % val = 3:3:33;
% matrix_size = [1280,720]; %[640,480];
% BG = "constBG";

%%

addpath("PERFORMANCE_METRICS\metrics_calc_functions\")
% addpath("..\event_stream\matlab\")

FullTestName = 'frequency_amplitude_heatmap';
vec = {'4','8','12','16','20','24'};
Tests = struct('name','','folder','','date','','bytes',0,'isdir',0,'datenum',0);
ci = 1;
for vi =1:length(vec)
    if ~isempty(dir(['OUTPUT\',FullTestName,'*_',vec{vi},'*']))
        Tests(ci) = dir(['OUTPUT\',FullTestName,'*_',vec{vi},'*']);
        ci = ci+1;
    end
end
% FullTestName = 'T1';
% Tests = dir(['OUTPUT\',FullTestName,'*']);
epochs = 5;

% Inisilized full run metrics maps
BW_SNR_all = cell(length(Tests),1);
RSNR_all =  cell(length(Tests),1);
Al_RSNR_all =  cell(length(Tests),1);
leg = [];


for ti = 1:length(Tests)
    disp(' ')
    disp(['<< Working on results from ',Tests(ti).name,' >>']);
    config_data_file = dir(['OUTPUT\',Tests(ti).name,'\*as_run.ini']);
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

        % check that simulation gave enough result files
        count_files = length(dir(['OUTPUT\',Tests(ti).name,'\events_and_labels\*.mat']));
        if count_files<(epochs*length(vector))
            warning(['missing result files for test ',Tests(ti).name]);
        end


        % Inisilized metrics data for this test
        BW_SNR = zeros(1,length(vector));
        RSNR =  zeros(1,length(vector));
        Al_RSNR = zeros(1,length(vector));


        for vi = 1:length(vector)
            n_ep = 0;
            for ep = 1:epochs
                if ~mod(vector(vi),1)
                    add_0 = '.0';
                else
                    add_0 = '';
                end

                % load all event simulation results
                ev_file_name = ['OUTPUT\',Tests(ti).name,'\events_and_labels\ev_',test_data.InitParams.sim_name,'_',sanned_param{2},'_',num2str(vector(vi)),add_0,'_ep',num2str(ep),'.txt'];
                simdata_file_name = ['OUTPUT\',Tests(ti).name,'\events_and_labels\simdata_',test_data.InitParams.sim_name,'_',sanned_param{2},'_',num2str(vector(vi)),add_0,'_ep',num2str(ep),'.mat'];
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
                disp([sanned_param{2},' = ', num2str(vector(vi)),', ep',num2str(ep),' results:']);

                mean_Signal_event_rate = Signal_event_count./target_time;
                mean_BG_event_rate = Bg_event_count./(T-target_time);
                inds = ~isnan(mean_Signal_event_rate);
                bw_snr = sum(abs(mean_Signal_event_rate(inds)-mean_BG_event_rate(inds)));
                disp([' - BW_SNR = ',num2str(bw_snr)]);
                BW_SNR(vi) = BW_SNR(vi) + bw_snr; % average value
                % Add some sort of dysplay here for debuging

                % Calculate Rate SNR metric - TO FIX. Not a good metric...
                [rsnr, RateImage_Sig_med, RateImage_BG_med]= calc_RSNR(signal_rate_stack(:,:,1),bg_rate_stack(:,:,1),matrix_size);
                disp([' - Rate SNR = ', num2str(rsnr)])
                RSNR(vi) = RSNR(vi) + rsnr;
                % figure;
                % subplot(2,2,1)
                % imagesc(log(RateImage_Sig_med+1)'); colorbar ;
                % subplot(2,2,2)
                % imagesc(log(RateImage_BG_med+1)'); colorbar;
                
                % Align events according to target motion in frame
                [all_events_aligned,filtered_inds,target_time_al] = align_Events(all_events,simulation_data,matrix_size);
                target_time_al = target_time_al'*dt;
                sig_aligned_ind = sig_ind(filtered_inds);
                [all_rate_aligned_stack,signal_rate_aligned_stack,bg_rate_aligned_stack] = create_rate_image(all_events_aligned,matrix_size,[],sig_aligned_ind);

                % Calculate Rate SNR metric for aligned event cloud - TO FIX. Not a good metric...
                [al_rsnr, RateImage_Sig_aligned_med, RateImage_BG_aligned_med]= calc_RSNR(signal_rate_aligned_stack(:,:,1),bg_rate_aligned_stack(:,:,1),matrix_size);
                disp([' - aligned Rate SNR = ', num2str(al_rsnr)])
                Al_RSNR(vi) = Al_RSNR(vi) + al_rsnr;

                % subplot(2,2,3)
                % imagesc(log(RateImage_Sig_aligned_med+1)'); colorbar ;
                % title([' V=',num2str(val(vi)), 'SIG - aligned'])
                % subplot(2,2,4)
                % imagesc(log(RateImage_BG_aligned_med+1)'); colorbar;
                % title(['BG. RSNR=', num2str(Al_RSNR(psft,vi))])

            end
        end

        BW_SNR_all{ti} = BW_SNR/n_ep;
        RSNR_all{ti} = RSNR/n_ep;
        Al_RSNR_all{ti} = Al_RSNR/n_ep;

        figure(1)
        loglog(vector,BW_SNR_all{ti}); hold on; grid on;
        leg = [leg,{replace(Tests(ti).name,'_',' ')}];
        xlabel(replace(sanned_param{2},'_',' '));
        ylabel('BW_S_N_R [Hz]')

        figure(2)
        loglog(vector,RSNR_all{ti}); hold on; grid on;
        xlabel(replace(sanned_param{2},'_',' '));
        ylabel('Rate SNR')

        figure(3)
        loglog(vector,Al_RSNR_all{ti}); hold on; grid on;
        xlabel(replace(sanned_param{2},'_',' '));
        ylabel('Aligned Rate SNR')
        drawnow
        
        disp(['Done evaluating results from ',Tests(ti).name]);
        disp(' ');
    end
end
figure(1)
legend(leg)
figure(2)
legend(leg)
figure(3)
legend(leg)
% figure;
% plot(val-max(val)/2,RSNR); hold on
% plot(val-max(val)/2,Al_RSNR); legend('not aligned RateSNR','aligned RateSNR');
% 
% figure;
% plot(val-max(val)/2,BW_SNR); %hold on
% plot(val-max(val)/2,Al_BWSNR); legend('not aligned RateSNR','aligned RateSNR');

