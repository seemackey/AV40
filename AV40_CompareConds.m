%% Compare attention conditions in AV40 data
% chase m 2025

clear
clc
close all
% Epoched file info
aud_dir = '/Volumes/Samsung03/data/AV40/Peter/pt035/imported/Aud/';
vis_dir = '/Volumes/Samsung03/data/AV40/Peter/pt035/imported/Vis/';
audfile = 'pt035000015_analog_imported.mat'; % epoched file
audpath = fullfile(aud_dir,audfile);
visfile = 'pt035000016_analog_imported.mat'; 
vispath = fullfile(vis_dir,visfile);

load(audpath);
epoched_data_aud = epoched_data;
clear epoched_data
load(vispath);
epoched_data_vis = epoched_data;




fs = config.newadrate; % Sampling rate (downsampled)
epochSamples = size(epoched_data_aud.standard.csd, 3); % Number of time points in each epoch
epochTime = linspace(config.epoch_tframe(1), config.epoch_tframe(2), epochSamples); % Time vector (ms)

selectedChannels = [2];


% Compute mean & CI per selected channel
for i = 1:length(selectedChannels)
    ch = selectedChannels(i);

    % Extract LFP epochs (channels x trials x time)
    lfp1 = squeeze(epoched_data_aud.standard.mua(ch, :, :)); % Attend Auditory
    lfp2 = squeeze(epoched_data_vis.standard.mua(ch, :, :)); % Attend Visual

    % Compute Mean & CI
    mean1(i, :) = nanmean(lfp1, 1);
    ci1(i, :) = nanstd(lfp1, [], 1) ./ sqrt(size(lfp1, 1)) * 1.96;

    mean2(i, :) = nanmean(lfp2, 1);
    ci2(i, :) = nanstd(lfp2, [], 1) ./ sqrt(size(lfp2, 1)) * 1.96;
end

figular = figure;
figular.Position = [150 150 700 1800];
for i = 1:length(selectedChannels)
    subplot(length(selectedChannels), 1, i);
    hold on;

    % Plot mean ± CI for Attend Auditory
    fill([epochTime, fliplr(epochTime)], [mean1(i, :) - ci1(i, :), fliplr(mean1(i, :) + ci1(i, :))], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none','HandleVisibility','off');
    plot(epochTime, mean1(i, :), 'b', 'LineWidth', 2);
    

    % Plot mean ± CI for Attend Visual
    fill([epochTime, fliplr(epochTime)], [mean2(i, :) - ci2(i, :), fliplr(mean2(i, :) + ci2(i, :))], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none','HandleVisibility','off');
    plot(epochTime, mean2(i, :), 'r', 'LineWidth', 2);
    
    % plot diff
    %plot(epochTime, mean1(i,:)-mean2(i, :), 'r', 'LineWidth', 2);
    
    % Formatting
    set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'GridLineStyle', '--');
    xlabel('Time (ms)', 'FontSize', 12);
    ylabel('CSD amp', 'FontSize', 12);
    title(['Channel ', num2str(selectedChannels(i))], 'FontSize', 14);
    %grid on;
    legend('Attend Aud','Attend V','Location','Best')
end