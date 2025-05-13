function plot_baseline_corrected_data_mult_trigType(epoched_data, config, figuresDir, fileName)
 % Adapted to handle the new data structure with trigType grouping
disp(['oo you are fancy; Making figures for mult trig types']);

% Define baseline correction period
baselineStart = -30;  
baselineEnd = -3;     

% Ensure baseline period is within the epoch timeframe
if baselineStart < config.epoch_tframe(1)
    warning('Baseline start is outside the epoch range. Using start of epoch.');
    baselineStart = config.epoch_tframe(1);
end
if baselineEnd > config.epoch_tframe(2)
    warning('Baseline end is outside the epoch range. Using end of epoch.');
    baselineEnd = config.epoch_tframe(2);
end

epochTimeframe = config.epoch_tframe(1):config.epoch_tframe(2); 
fs = config.newadrate;

% Find sample indices corresponding to baseline range
[~, baselineStartIdx] = min(abs(epochTimeframe - baselineStart));
[~, baselineEndIdx] = min(abs(epochTimeframe - baselineEnd));
baselineIdx = baselineStartIdx:baselineEndIdx;

% Extract unique trigTypes
uniqueTrigTypes = unique([epoched_data.sortedData.trigType]);

for tt = 1:length(uniqueTrigTypes)
    trigType = uniqueTrigTypes(tt);
    disp(['Processing trigType: ', num2str(trigType)]);

    % Find indices for this trigType
    if ~isnan(trigType)
        indices = find([epoched_data.sortedData.trigType] == trigType);
    else
        
        indices = tt; % for some reason this one broke my brain
    end

    % Collect baseline-corrected data across all indices of this trigType
    lfp_all = [];
    csd_all = [];
    mua_all = [];

    for idx = indices
        data = epoched_data.sortedData(idx);

        % Initialize arrays for this particular entry
        lfp_bsl = zeros(size(data.lfp));
        csd_bsl = zeros(size(data.csd));
        if ~isempty(data.mua)
            mua_bsl = zeros(size(data.mua));
        else
            mua_bsl = [];
        end

        % artifact reject
            [epoched_data.lfp, ~] = MTF_rejectartifacts(data.lfp, 'median', 3);
            [epoched_data.csd, ~] = MTF_rejectartifacts(data.csd, 'median', 3);
            [epoched_data.mua, ~] = MTF_rejectartifacts(data.mua, 'median', 3);


        % Baseline Correction for each channel and trial
        for chct = 1:size(data.lfp, 1)
            for trct = 1:size(data.lfp, 2)
                % LFP Baseline Correction
                lfp_bsl(chct, trct, :) = squeeze(data.lfp(chct, trct, :)) - ...
                    mean(squeeze(data.lfp(chct, trct, baselineIdx)), 'omitnan');

                % CSD Baseline Correction
                csd_bsl(chct, trct, :) = squeeze(data.csd(chct, trct, :)) - ...
                    mean(squeeze(data.csd(chct, trct, baselineIdx)), 'omitnan');

                % MUA Baseline Correction (if available)
                if ~isempty(mua_bsl)
                    mua_bsl(chct, trct, :) = squeeze(data.mua(chct, trct, :)) - ...
                        mean(squeeze(data.mua(chct, trct, baselineIdx)), 'omitnan');
                end
            end
        end

        % Concatenate the baseline-corrected data
        lfp_all = cat(2, lfp_all, lfp_bsl);
        csd_all = cat(2, csd_all, csd_bsl);
        if ~isempty(mua_bsl)
            mua_all = cat(2, mua_all, mua_bsl);
        end
    end

    % Trial averaging across the second dimension (trials)
    lfp_avg = mean(lfp_all, 2, 'omitnan');
    csd_avg = mean(csd_all, 2, 'omitnan');
    if ~isempty(mua_all)
        mua_avg = mean(mua_all, 2, 'omitnan');
    else
        mua_avg = [];
    end

    % Plotting parameters
    timeVector = linspace(config.epoch_tframe(1), config.epoch_tframe(2), size(lfp_avg, 3));
    numChannels = 1:size(lfp_avg, 1);
    numChannelsCSD = 1:size(csd_avg, 1);

    % Determine symmetrical color axis
    lfp_min = min(lfp_avg, [], 'all');
    lfp_max = max(lfp_avg, [], 'all');
    lfp_caxis = [-max(abs([lfp_min, lfp_max])), max(abs([lfp_min, lfp_max]))];

    csd_min = min(csd_avg, [], 'all');
    csd_max = max(csd_avg, [], 'all');
    csd_caxis = [-max(abs([csd_min, csd_max])), max(abs([csd_min, csd_max]))];

    if ~isempty(mua_avg)
        mua_min = min(mua_avg, [], 'all');
        mua_max = max(mua_avg, [], 'all');
        mua_caxis = [-max(abs([mua_min, mua_max])), max(abs([mua_min, mua_max]))];
    end

    % Plot Figures
    fig = figure('Position', [200, 200, 1200, 700]);

    % Plot LFP
    subplot(1, 3, 1);
    imagesc(timeVector, numChannels, squeeze(lfp_avg));
    title(['Trial Avg. LFP (' ', TrigType ', num2str(trigType), ')']);
    xlabel('Time (ms)');
    ylabel('Channel');
    colormap(flipud(jet));
    colorbar;
    caxis(lfp_caxis);

    % Plot CSD
    subplot(1, 3, 2);
    imagesc(timeVector, numChannelsCSD, squeeze(csd_avg));
    title(['Trial Avg. CSD (' ', TrigType ', num2str(trigType), ')']);
    xlabel('Time (ms)');
    ylabel('Channel');
    colorbar;
    caxis(csd_caxis);

    % Plot MUA (if available)
    if ~isempty(mua_avg)
        subplot(1, 3, 3);
        imagesc(timeVector, numChannelsCSD, squeeze(mua_avg));
        title(['Trial Avg. MUA (' ', TrigType ', num2str(trigType), ')']);
        xlabel('Time (ms)');
        ylabel('Channel');
        colorbar;
        caxis(mua_caxis);
    end

    % Save Figures
    saveas(fig, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType', num2str(trigType), '_avg_profiles.fig']));
    saveas(fig, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType', num2str(trigType), '_avg_profiles.jpg']));
    close(fig);

    disp(['Baseline correction and plots for trigType ', num2str(trigType), ' have been saved successfully.']);

    % save mean
    durationForTuningCurve = 100;
    if ~isempty(mua_avg)
        tuningcurve(tt,:) = mean(mua_avg(:,baselineEndIdx:durationForTuningCurve),2);
    end
    
    
end

% Generate average MUA data matrix (e.g., tuningcurve)
% Assume tuningcurve is of size [numTrigTypes, numChannels]
% Where rows = trigger types (e.g., tone frequencies), columns = channels
for norm_count = 1:size(tuningcurve,2)
    tuningcurve(:,norm_count) = tuningcurve(:,norm_count)/max(tuningcurve(:,norm_count));
end


numTrigTypes = size(tuningcurve, 1);
numChannels = size(tuningcurve, 2);


% Plot MUA Mean for Each Channel as a Function of TrigType
figMUA = figure('Position', [200, 200, 1200, 700]);

hold on;
% Determine a dynamic offset that avoids overlap
channelRanges = max(tuningcurve, [], 1) - min(tuningcurve, [], 1);  % range per channel
%offset = max(channelRanges) * 1.5;  % buffer to avoid overlaps
offset = 0.4;

colors = lines(size(tuningcurve, 2));  % Generate unique colors for each channel
numChannels = size(tuningcurve, 2);

% Generate YData to mimic imagesc behavior: Channel 1 at top, Channel N at bottom
yData = fliplr(1:numChannels);  % Reversing the order of channels

% Plot each channel, but assign them to a reversed y-axis order
for ch = 1:numChannels
    % Calculate the offset for each channel in reversed order
    y_offset = find(yData == ch) * offset;  % Find where the current channel would be plotted on the flipped y-axis
    plot(1:length(uniqueTrigTypes), tuningcurve(:,ch) + y_offset, 'Color', colors(ch,:),'LineWidth',1);
end

% Add labels and title
set(gca, 'LineWidth', 1.5);  %thicccc
set(gca, 'YDir', 'normal');  
%set(gca, 'XScale', 'log')

title('Mean MUA');
%xlabel('Tone Freq (kHz)');
%xlabel('Mod Freq')
ylabel('Channels (Offset for Clarity)');
set(gca,'FontSize',16)

% Set Y-ticks to match the desired order (Ch 1 at the top, Ch N at the bottom)
yticks((1:numChannels) * offset);
yticklabels(arrayfun(@(x) sprintf('Ch %d', x), yData, 'UniformOutput', false));

xticks(1:length(uniqueTrigTypes));  % Make sure xticks match your trigger types

if nanmean(uniqueTrigTypes) > 300
    xticklabels({'0.3','0.5','0.7','1','1.4','2','2.8','4','5.6','8','11','16','22','32','Noise'});
end
    
grid on;
hold off;

% Save Figures
saveas(figMUA, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType_MUA_TuningCurve.fig']));
saveas(figMUA, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType_MUA_TuningCurve.jpg']));
end