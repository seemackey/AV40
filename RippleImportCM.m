% import ephys from the ripple system
% chase m 2024


% Clear workspace and initialize
clear;
close all;
clc;

% Directories and file names

inputDir = 'H:\data\AV40\Peter'; % Replace with your input directory
figuresDir = 'H:\data\AV40\Peter\imported'; % Replace with your output directory
fileName = 'pt027000029.nev'; % Replace with your file name

% Example configuration
config = struct();
config.epoch_tframe = [-50, 150]; % Epoch window in ms
config.newadrate = 1000;          % Resampling rate
config.filters.lfp = [0.5, 300];  % LFP filter range (Hz)
config.filters.mua = [300, 5000]; % MUA filter range (Hz)
config.trigger_channel = 93;  %  analog trigger channel for aud
config.channels = [1:24];             % ephys data channels
config.channel_remap = true; % Enable channel remapping by default
config.trigger_method = 'analog'; % 'events' or 'analog'
config.trigger_threshold = 50;   % Threshold for analog trigger detection
config.event_entity_id = 1;       % Default Event Entity ID
config.artifact_threshold = 3;    % Z-score threshold for artifact rejection
config.checksync = 0; % check sync between ripple and eyelink
config.get_deviant = 1;

try
    % Call the main data import function
    data_import_v2(inputDir,figuresDir, fileName, config);
    fprintf('importing\n');
catch ME
    % Handle errors 
    fprintf('An error occurred: %s\n', ME.message);
end

function data_import_v2(directory1, figuresDir,fileName, config)
    % DATA_IMPORT_V2 - Enhanced Import Script with Raw and Analog Data Handling
    %
    % Parameters:
    % directory1: Directory containing the raw data files.
    % fileName: Name of the file to process.
    % config: Configuration structure with user-defined parameters.

    % Default Configuration
    if nargin < 3
        config = get_default_config();
    end

    % Open File
    filePath = fullfile(directory1, fileName);
    [ns_RESULT, hFile] = ns_OpenFile(filePath);
    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Failed to open file: %s', ns_RESULT);
    end

    % Separate entities into raw (data) and analog (stimuli and eye mvmt) categories
    [rawChannels, ~] = classify_channels(hFile);

    % Handle triggers
    if strcmp(config.trigger_method, 'analog')
        triggerChannel = config.trigger_channel; % Specify analog channel for triggers
        [triggers_std,triggers_deviant] = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold);
    else %digital triggers don't work yet, sorrrryyyyyy
        disp('no analog triggers? check analog trig chan')
        % digital trig logic could go here
    end

    
    % get the eyelink data
    timingResults = AV40_importEyelink(directory1, fileName);
    
    %% check match between eyelink ripple and eyelink trigs

if config.checksync ==1

% Assumptions:
% - `triggers`: Analog trigger timestamps (samples at 30 kHz)
% - `timingResults.ADDT_STANDARD`: Timing results ADDT standard event timestamps (ms at 1 kHz)
% - `timingResults.ADDT_DEVIANT`: Timing results ADDT deviant event timestamps (ms at 1 kHz)
% - `fs_analog`: Sampling rate of analog triggers (e.g., 30000 Hz)

% Constants
    fs_analog = 30000; % Sampling rate of analog triggers (30 kHz)
    fs_timing = 1000;  % Sampling rate of timingResults (1 kHz)

    %% 1. Downsample Analog Triggers to 1 kHz
    disp('Downsampling analog triggers to 1 kHz...');
    triggers_downsampled = triggers / (fs_analog / fs_timing); % Convert to 1 kHz sample indices


    %% 2. Combine ADDT Standards and Deviants
    
    combined_ADDT = sort([timingResults.ADDT_STANDARD, timingResults.ADDT_DEVIANT]);

    % Ensure enough triggers and combined timestamps for comparison
    numSamples = min([100, length(triggers_downsampled), length(combined_ADDT)]);
    if numSamples < 100
        warning('Fewer than 100 triggers available for comparison!');
    end

    % Select subsets for comparison
    analog_subset = triggers_downsampled(1:numSamples);
    timing_subset = combined_ADDT(1:numSamples);

    %% 3. Align Both Trigger Sets to their First Event
    disp('Aligning triggers relative to their first event...');

    % Align relative to the first trigger
    analog_relative = analog_subset - analog_subset(1);
    timing_relative = timing_subset - timing_subset(1);

    % Verify alignment
    disp('First Trigger (Analog, Timing):');
    fprintf('Analog: %.2f ms | Timing: %.2f ms\n', analog_subset(1), timing_subset(1));

    %% 4. Calculate Relative Differences
    disp('Calculating relative differences...');

    % Compute differences between corresponding relative timestamps
    relative_diff = analog_relative - timing_relative;

    % Compute mean and standard deviation of relative differences
    mean_relative_diff = mean(relative_diff);
    std_relative_diff = std(relative_diff);

    fprintf('Mean Relative Difference: %.2f ms\n', mean_relative_diff);
    fprintf('Std Relative Difference: %.2f ms\n', std_relative_diff);

    %% 5. Calculate Interstimulus Intervals (ISIs) in Relative Time
    disp('Calculating ISIs for relative timing...');

    % Calculate ISIs (relative time differences)
    ISI_analog = diff(analog_relative);
    ISI_timing = diff(timing_relative);

    % Compute mean and std ISIs
    ISI_stats = struct();
    ISI_stats.analog.mean = mean(ISI_analog);
    ISI_stats.analog.std = std(ISI_analog);
    ISI_stats.timing.mean = mean(ISI_timing);
    ISI_stats.timing.std = std(ISI_timing);

    disp('ISI Statistics (Relative Time):');
    disp(ISI_stats);

    %% 6. Plot Results
    disp('Plotting relative timing and ISI comparison...');

    figure;

    % Plot Relative Timestamps
    subplot(2,1,1);
    plot(analog_relative, '-o', 'DisplayName', 'Analog Triggers (Relative)');
    hold on;
    plot(timing_relative, '-x', 'DisplayName', 'Combined ADDT (Relative)');
    legend('Location', 'best');
    title('Relative Trigger Timing Comparison (Analog vs Combined ADDT)');
    xlabel('Trigger Index');
    ylabel('Time (ms)');
    grid on;

    % Plot Relative Differences
    subplot(2,1,2);
    plot(relative_diff, '-o');
    yline(mean_relative_diff, '--r', 'Mean Diff');
    yline(mean_relative_diff + std_relative_diff, ':r', 'Mean + Std');
    yline(mean_relative_diff - std_relative_diff, ':r', 'Mean - Std');
    title('Relative Trigger Time Differences (Analog vs Combined ADDT)');
    xlabel('Trigger Index');
    ylabel('Difference (ms)');
    grid on;

    %% 7. Display Summary
    disp('Summary of Relative Trigger Comparison:');
    fprintf('Mean Relative ISI (Analog): %.2f ms\n', ISI_stats.analog.mean);
    fprintf('Mean Relative ISI (Combined ADDT): %.2f ms\n', ISI_stats.timing.mean);
    fprintf('Mean Relative Trigger Difference: %.2f ms\n', mean_relative_diff);
    fprintf('Std Relative Trigger Difference: %.2f ms\n', std_relative_diff);


end
    %%
    % get ephys Data
    channels = config.channels;
    rawData = get_channel_data(hFile, channels);
    
    

    % Check if remapping is needed
    if isfield(config, 'channel_remap') == 1
        rawData = remap_channels(rawData);
    end

    


    % Filter Raw Data
    filtertype=1;
    disp('filtering')
    newadrate = config.newadrate;
    filtere = config.filters.lfp;
    filterm = config.filters.mua;
    adrate = 30000; %ASSUMED for ripple analog fs
    
    [cnte, cntm, cntc, ~, ~] = module_filtcont(rawData, newadrate, filtere, filterm, filtertype,adrate);

    % Epoch Data to aud triggers in ripple system
    disp('epoching')
    triggers_std = triggers_std / (adrate / newadrate); 
    triggers_deviant = triggers_deviant / (adrate / newadrate);
    epoched_data = struct();
    epoched_data.standard = [];
    epoched_data.deviant = [];
    [lfp_std, mua_std, csd_std] = epoch_data(cnte, cntm, cntc, triggers_std, config);
    % Reject Artifacts
    [lfp_std, mua_std, csd_std, triggers_std] = reject_artifacts(lfp_std, mua_std, csd_std, triggers_std, config);
    
    epoched_data.standard.lfp = lfp_std;
    epoched_data.standard.mua = mua_std;
    epoched_data.standard.csd = csd_std;
    
    if config.get_deviant == 1
    [lfp_dev, mua_dev, csd_dev] = epoch_data(cnte, cntm, cntc, triggers_deviant, config);
    % Reject Artifacts
    [lfp_dev, mua_dev, csd_dev, triggers_deviant] = reject_artifacts(lfp_dev, mua_dev, csd_dev, triggers_deviant, config);
    
    epoched_data.deviant.lfp = lfp_dev;
    epoched_data.deviant.mua = mua_dev;
    epoched_data.deviant.csd = csd_dev;
    end
    

    % Save Results
    disp('saving data')
    save_results(figuresDir, fileName, epoched_data, triggers_std,triggers_deviant,timingResults,config);
    
    %% make figs
    disp('making figures')
    % Define directories for saving figures
    figuresDir = fullfile(directory2, 'figures');
    if ~exist(figuresDir, 'dir')
        mkdir(figuresDir);
    end

    % Plot Standard Data
    plot_baseline_corrected_data(epoched_data.standard, config, 'standard', figuresDir, fileName);

    % Plot Deviant Data (if available)
    if isfield(epoched_data, 'deviant') && ~isempty(epoched_data.deviant)
        plot_baseline_corrected_data(epoched_data.deviant, config, 'deviant', figuresDir, fileName);
    end



    
end

function plot_baseline_corrected_data(epoched_data, config, condition, figuresDir, fileName)
    % PLOT_BASELINE_CORRECTED_DATA - Baseline correct and visualize epoched data
    %
    % Parameters:
    % epoched_data - Struct containing LFP, CSD, MUA data
    % config       - Configuration struct with epoch and sampling settings
    % condition    - String: 'standard' or 'deviant' for labeling
    % figuresDir   - Directory to save figures
    % fileName     - Base name for saving figures
    
    disp(['Making figures for ', condition, ' data']);

    %% Baseline Correction Parameters
    baselineStart = config.epoch_tframe(1); % Start of baseline window (ms)
    baselineEnd = -5; % End of baseline window (ms)
    fs = config.newadrate; % Sampling rate (Hz)

    % Convert baseline window from ms to sample indices
    baselineIdx = round((baselineStart:baselineEnd) * (fs / 1000)) - baselineStart * (fs / 1000) + 1;

    %% Baseline Correction
    % Initialize baseline-corrected arrays
    lfp_bsl = zeros(size(epoched_data.lfp));
    csd_bsl = zeros(size(epoched_data.csd));
    mua_bsl = zeros(size(epoched_data.mua));

    % Baseline Correction for LFP
    for trct = 1:size(epoched_data.lfp, 2)
        for chct = 1:size(epoched_data.lfp, 1)
            lfp_bsl(chct, trct, :) = squeeze(epoched_data.lfp(chct, trct, :)) - ...
                mean(squeeze(epoched_data.lfp(chct, trct, baselineIdx)), 'omitnan');
        end
    end

    % Baseline Correction for CSD
    for trct = 1:size(epoched_data.csd, 2)
        for chct = 1:size(epoched_data.csd, 1)
            csd_bsl(chct, trct, :) = squeeze(epoched_data.csd(chct, trct, :)) - ...
                mean(squeeze(epoched_data.csd(chct, trct, baselineIdx)), 'omitnan');
        end
    end

    % Baseline Correction for MUA
    if ~isempty(epoched_data.mua)
        for trct = 1:size(epoched_data.mua, 2)
            for chct = 1:size(epoched_data.mua, 1)
                mua_bsl(chct, trct, :) = squeeze(epoched_data.mua(chct, trct, :)) - ...
                    mean(squeeze(epoched_data.mua(chct, trct, baselineIdx)), 'omitnan');
            end
        end
    end

    %% Plotting Parameters
    timeVector = linspace(config.epoch_tframe(1), config.epoch_tframe(2), size(lfp_bsl, 3));
    numChannels = 1:size(lfp_bsl, 1);

    % Determine Color Axis for CSD
    csd_min = min(min(squeeze(mean(csd_bsl(:, :, :), 2))));
    csd_max = max(max(squeeze(mean(csd_bsl(:, :, :), 2))));
    csd_caxis = [-max(abs([csd_min, csd_max])) * 0.75, max(abs([csd_min, csd_max])) * 0.75];

    % Determine Color Axis for MUA (if available)
    if ~isempty(mua_bsl)
        mua_min = min(min(squeeze(mean(mua_bsl(:, :, :), 2))));
        mua_max = max(max(squeeze(mean(mua_bsl(:, :, :), 2))));
        mua_caxis = [-max(abs([mua_min, mua_max])) * 0.75, max(abs([mua_min, mua_max])) * 0.75];
    end

    %% Plot Figures
    fig = figure('Position', [200, 200, 1200, 700]);

    % Plot LFP
    subplot(1, 3, 1);
    imagesc(timeVector, numChannels, squeeze(mean(lfp_bsl, 2)));
    title(['Trial Avg. LFP (', condition, ')']);
    xlabel('Time (ms)');
    ylabel('Channel');
    colormap(flipud(jet));
    colorbar;

    % Plot CSD
    subplot(1, 3, 2);
    imagesc(timeVector, numChannels, squeeze(mean(csd_bsl, 2)));
    title(['Trial Avg. CSD (', condition, ')']);
    xlabel('Time (ms)');
    ylabel('Channel');
    colorbar;
    caxis(csd_caxis);

    % Plot MUA (if available)
    if ~isempty(mua_bsl)
        subplot(1, 3, 3);
        imagesc(timeVector, numChannels, squeeze(mean(mua_bsl, 2)));
        title(['Trial Avg. MUA (', condition, ')']);
        xlabel('Time (ms)');
        ylabel('Channel');
        colorbar;
        caxis(mua_caxis);
    end

    %% Save Figures
    saveas(fig, fullfile(figuresDir, [fileName, '_', condition, '_avg_profiles.fig']));
    saveas(fig, fullfile(figuresDir, [fileName, '_', condition, '_avg_profiles.jpg']));
    close(fig);

    disp(['Baseline correction and plots for ', condition, ' have been saved successfully.']);
end

function rawData = remap_channels(rawData)
    % REMAP_CHANNELS - Adjusts raw data channels to correct hardware mismatch
    %
    % Parameters:
    % rawData: Original raw electrophysiology data matrix (channels x samples)
    %
    % Output:
    % rawData: Remapped raw data with corrected channel alignment

    fprintf('Applying channel remapping to correct alignment mismatch...\n');
    X = 1:2:size(rawData, 1) - 1; % Odd indices
    X1 = 2:2:size(rawData, 1);    % Even indices

    % Swap channels
    remappedData = rawData; % Copy original data
    remappedData(X, :) = rawData(X1, :); % Odd -> Even
    remappedData(X1, :) = rawData(X, :); % Even -> Odd

    rawData = remappedData;
    disp('done remapping')
end


function [rawChannels, analogChannels] = classify_channels(hFile)
    % Classify channels as raw or analog based on labels
    rawChannels = [];
    analogChannels = [];
    numEntities = length(hFile.Entity);

    for i = 1:numEntities
        label = hFile.Entity(i).Label;
        if contains(label, 'raw', 'IgnoreCase', true)
            rawChannels(end + 1) = i;
        elseif contains(label, 'analog', 'IgnoreCase', true)
            analogChannels(end + 1) = i;
        end
    end
end

function data = get_channel_data(hFile, channels)
    % GET_CHANNEL_DATA - Retrieve data for specified channels sequentially
    %
    % Parameters:
    % hFile: Handle to the Neuroshare file
    % channels: Array of channel IDs to import
    %
    % Output:
    % data: Matrix (channels x samples) of retrieved analog data
    
    data = [];
    for ch = channels
        [~, ~, channelData] = ns_GetAnalogData(hFile, ch, 1, 1e8);
        data = [data; channelData']; %#ok<AGROW> Transpose for consistency
    end
end





function [triggers_standard, triggers_deviant] = get_analog_triggers(hFile, analogChannel, threshold)
    % GET_ANALOG_TRIGGERS - Detect first pulse of each pulse train in analog data
    %
    % Parameters:
    % hFile          - Handle to the Neuroshare file
    % analogChannel  - Analog channel ID
    % threshold      - Threshold for detecting rising edges
    %
    % Outputs:
    % triggers_standard - Indices of the first pulse of standard trains (~5 pulses)
    % triggers_deviant  - Indices of the first pulse of deviant trains (~10 pulses)
    
    %% Step 1: Retrieve Analog Data
    [ns_RESULT, ~, analogData] = ns_GetAnalogData(hFile, analogChannel, 1, hFile.Entity(analogChannel).Count);

    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Error retrieving analog data: %s', ns_RESULT);
    end

    % Define sampling rate
    samplingRateStr = '30 ksamp/sec';
    if strcmp(samplingRateStr, '30 ksamp/sec')
        fs = 30000; % Sampling rate
        disp('Found 30 ksamp/sec! the bar for success is low');
    else
        fs = 30000;
        disp('check yourself before you wreck yourself. Sampling rate may differ from 30 ksamp/sec.');
    end

    %% Step 2: Detect All Rising Edges
    allTriggers = find(diff(analogData > threshold) == 1); % Rising edge detection

    %% Step 3: Separate Pulse Trains
    minPulseGap = 0.15 * fs; % Minimum gap to separate pulse trains
    pulseTrainStarts = [];
    pulseTrainEnds = [];
    lastPulse = -inf; % Track the last pulse's index

    % Identify train start and end points
    for i = 1:length(allTriggers)
        if allTriggers(i) - lastPulse > minPulseGap
            % New train detected
            pulseTrainStarts(end + 1) = allTriggers(i); %#ok<AGROW>
            if i > 1
                pulseTrainEnds(end + 1) = allTriggers(i - 1); %#ok<AGROW>
            end
        end
        lastPulse = allTriggers(i);
    end
    % Add final train end
    pulseTrainEnds(end + 1) = allTriggers(end);

    %% Step 4: Classify Trains as Standard or Deviant
    triggers_standard = [];
    triggers_deviant = [];

    for i = 1:length(pulseTrainStarts)
        % Extract pulses within this train
        startIdx = find(allTriggers >= pulseTrainStarts(i), 1, 'first');
        endIdx = find(allTriggers <= pulseTrainEnds(i), 1, 'last');
        trainPulses = allTriggers(startIdx:endIdx);

        % Count number of pulses in the train
        pulseCount = length(trainPulses);

        if pulseCount >= 8 % Deviant trains (~10 pulses)
            triggers_deviant(end + 1) = pulseTrainStarts(i); %#ok<AGROW>
        elseif pulseCount >= 4 && pulseCount <= 7 % Standard trains (~5 pulses)
            triggers_standard(end + 1) = pulseTrainStarts(i); %#ok<AGROW>
        end
    end

    %% Step 5: Optional Plot for Visualization
%     figure;
%     plot(1:length(analogData), analogData);
%     hold on;
%     plot(triggers_standard, analogData(triggers_standard), 'go', 'MarkerSize', 8, 'LineWidth', 2);
%     plot(triggers_deviant, analogData(triggers_deviant), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
%     title('Analog Signal with Detected Standard and Deviant Triggers');
%     xlabel('Sample Index');
%     ylabel('Analog Signal');
%     legend('Analog Signal', 'Standard Triggers (~5 pulses)', 'Deviant Triggers (~10 pulses)');
%     hold off;
% 
%     disp(['Detected ', num2str(length(triggers_standard)), ' standard pulse trains and ', num2str(length(triggers_deviant)), ' deviant pulse trains.']);
end


function data = get_triggered_channel_data(hFile, channels, triggers, preTrigger, postTrigger, samplingRate)
    % GET_TRIGGERED_CHANNEL_DATA - Import channel data around trigger times
    %
    % Parameters:
    % hFile: Handle to the Neuroshare file
    % channels: Array of channel IDs to import
    % triggers: Array of trigger timestamps (in seconds)
    % preTrigger: Time (in seconds) before each trigger to include
    % postTrigger: Time (in seconds) after each trigger to include
    % samplingRate: Sampling rate of the analog data (Hz)
    %
    % Output:
    % data: 3D Matrix (channels x triggers x samples per epoch)

    fprintf('Importing data around triggers from %d channels...\n', length(channels));

    % Calculate the number of samples per epoch
    samplesPerEpoch = round((preTrigger + postTrigger) * samplingRate);
    numTriggers = length(triggers);
    numChannels = length(channels);
    
    % Pre-allocate data matrix
    data = NaN(numChannels, numTriggers, samplesPerEpoch);

    % Loop through each channel
    for chIdx = 1:numChannels
        ch = channels(chIdx);
        fprintf('Processing Channel %d...\n', ch);
        
        % Loop through each trigger
        for trigIdx = 1:numTriggers
            triggerTime = triggers(trigIdx);
            
            % Calculate sample indices
            startSample = round((triggerTime - preTrigger) * samplingRate) + 1;
            endSample = startSample + samplesPerEpoch - 1;
            
            % Ensure indices are valid
            if startSample < 1
                warning('Trigger %d on channel %d starts before data begins. Skipping...', trigIdx, ch);
                continue;
            end
            
            try
                % Retrieve the specific window of data
                [ns_RESULT, ~, epochData] = ns_GetAnalogData(hFile, ch, startSample, samplesPerEpoch);
                
                if ~strcmp(ns_RESULT, 'ns_OK')
                    warning('Failed to retrieve data for Trigger %d on Channel %d: %s', trigIdx, ch, ns_RESULT);
                    continue;
                end
                
                % Store data in pre-allocated array
                data(chIdx, trigIdx, :) = epochData';
                
            catch ME
                warning('Error processing Trigger %d on Channel %d: %s', trigIdx, ch, ME.message);
            end
        end
    end
    
    fprintf('Data import around triggers complete.\n');
end


function timingResults = AV40_importEyelink(dataDirectory, filename)
    % AV40_IMPORT_EYELINK - Import and parse TIMING_RESULTS and DEVIANT_RESPONSE_RESULTS
    %
    % inputs:
    % dataDirectory: Path to the main data dir containing edf subdir
    % filename: Name of the current data file (e.g., 'pt027000029.nev')
    %
    % Output:
    % timingResults: Struct containing extracted event times for specified events
    
    %% Step 1: Locate the 'edf' Subdirectory
    edfDir = fullfile(dataDirectory, 'edf');
    if ~isfolder(edfDir)
        error('The ''edf'' subdirectory does not exist in: %s', dataDirectory);
    end
    
    % Remove file extension
    [~, baseFilename, ~] = fileparts(filename);
    fprintf('Searching for subdirectory: %s in %s\n', baseFilename, edfDir);

    %% Step 2: Locate Matching Subdirectory
    subdirs = dir(edfDir);
    subdirs = subdirs([subdirs.isdir]); % Keep only directories
    matchingSubdir = '';
    
    for i = 1:length(subdirs)
        if strcmp(subdirs(i).name, baseFilename)
            matchingSubdir = fullfile(edfDir, subdirs(i).name);
            break;
        end
    end
    
    if isempty(matchingSubdir)
        error('No subdirectory matching "%s" was found in: %s', baseFilename, edfDir);
    end

    %% Step 3: Locate TIMING_RESULTS and DEVIANT_RESPONSE_RESULTS
    timingFilePath = fullfile(matchingSubdir, 'TIMING_RESULTS.txt');
    deviantFilePath = fullfile(matchingSubdir, 'DEVIANT_RESPONSE_RESULTS.txt');
    
    if ~isfile(timingFilePath)
        error('TIMING_RESULTS.txt not found in: %s', matchingSubdir);
    end
    if ~isfile(deviantFilePath)
        warning('DEVIANT_RESPONSE_RESULTS.txt not found in: %s', matchingSubdir);
        deviantFilePath = ''; % Allow proceeding without deviant file
    end
    
    %% Step 4: Parse TIMING_RESULTS
    fprintf('Parsing TIMING_RESULTS from: %s\n', timingFilePath);
    timingResults = struct();
    timingResults.ADDT_STANDARD = [];
    timingResults.ADDT_DEVIANT = [];
    timingResults.VDDT_STANDARD = [];
    timingResults.VDDT_DEVIANT = [];
    timingResults.VST_ONSET = [];
    
    fid = fopen(timingFilePath, 'rt');
    rawLines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    rawLines = rawLines{1};
    
    for i = 1:length(rawLines)
        line = strtrim(rawLines{i});
        if contains(line, 'ADDT_STANDARD_ONSET')
            timingResults.ADDT_STANDARD(end + 1) = extract_event_time(line);
        elseif contains(line, 'ADDT_DEVIANT_ONSET')
            timingResults.ADDT_DEVIANT(end + 1) = extract_event_time(line);
        elseif contains(line, 'VDDT_STANDARD_ONSET')
            timingResults.VDDT_STANDARD(end + 1) = extract_event_time(line);
        elseif contains(line, 'VDDT_DEVIANT_ONSET')
            timingResults.VDDT_DEVIANT(end + 1) = extract_event_time(line);
        elseif contains(line, 'VST_ONSET')
            timingResults.VST_ONSET(end + 1) = extract_event_time(line);
        end
    end
    
    %% Step 5: Parse DEVIANT_RESPONSE_RESULTS (if available)
    if ~isempty(deviantFilePath)
        fprintf('Parsing DEVIANT_RESPONSE_RESULTS from: %s\n', deviantFilePath);
        timingResults.DEVIANT_HIT_REWARD_JUICE_ONSET = [];
        timingResults.FALSE_ALARM_BUTTON_RESPONSE = [];
        timingResults.DEVIANT_MISS = [];
        
        fid = fopen(deviantFilePath, 'rt');
        rawLines = textscan(fid, '%s', 'Delimiter', '\n');
        fclose(fid);
        rawLines = rawLines{1};
        
        for i = 1:length(rawLines)
            line = strtrim(rawLines{i});
            if contains(line, 'DEVIANT_HIT_REWARD_JUICE_ONSET')
                timingResults.DEVIANT_HIT_REWARD_JUICE_ONSET(end + 1) = extract_event_time(line);
            elseif contains(line, 'FALSE_ALARM_BUTTON_RESPONSE')
                timingResults.FALSE_ALARM_BUTTON_RESPONSE(end + 1) = extract_event_time(line);
            elseif contains(line, 'DEVIANT_MISS')
                timingResults.DEVIANT_MISS(end + 1) = extract_event_time(line);
            end
        end
    else
        warning('Skipping DEVIANT_RESPONSE_RESULTS parsing (file missing).');
    end

    %% Step 6: Summary
%     fprintf('TIMING_RESULTS and DEVIANT_RESPONSE_RESULTS parsed successfully.\n');
%     disp(timingResults);


    %% Helper Function to Extract Event Time
    function eventTime = extract_event_time(line)
        % Extract EVENT_TIME from a line
        tokens = strsplit(line, '\t');
        eventTime = NaN;
        if length(tokens) >= 6
            eventTime = str2double(tokens{end});
            if isnan(eventTime)
                warning('Failed to extract EVENT_TIME from line: %s', line);
            end
        end
    end
end



function [eegLFP, eegMUA, eegCSD] = epoch_data(lfp, mua, csd, triggers, config)
    % Epoch data around triggers
    numTrials = length(triggers);
    numChannels = size(lfp, 1);
    epochLength = diff(config.epoch_tframe) * config.newadrate / 1000;

    eegLFP = zeros(numChannels, numTrials, epochLength);
    eegMUA = zeros(numChannels, numTrials, epochLength);
    eegCSD = zeros(numChannels, numTrials, epochLength);

    for trial = 1:numTrials
        idx = triggers(trial) + (config.epoch_tframe(1):config.epoch_tframe(2)) * config.newadrate / 1000;
        eegLFP(:, trial, :) = lfp(:, idx);
        eegMUA(:, trial, :) = mua(:, idx);
        eegCSD(:, trial, :) = csd(:, idx);
    end
end

function [eegLFP, eegMUA, eegCSD, triggers] = reject_artifacts(eegLFP, eegMUA, eegCSD, triggers, config)
    % Remove trials with artifacts
    [eegLFP, outliers] = MTF_rejectartifacts(eegLFP, 'median', config.artifact_threshold);
    eegMUA(:, outliers, :) = [];
    eegCSD(:, outliers, :) = [];
    triggers(outliers) = [];
end

function save_results(directory, fileName, epoched_data, triggers_std,triggers_deviant,eyelinkTiming,config)
    % Save processed data
    save(fullfile(directory, [fileName '_imported.mat']), 'epoched_data','triggers_std','triggers_deviant','eyelinkTiming','config');
end

function config = get_default_config()
    % Define default processing parameters
    config.epoch_tframe = [-50, 150]; % Epoch window in ms
    config.newadrate = 1000;          % Resampling rate
    config.filters.lfp = [0.5, 300];  % LFP filter range (Hz)
    config.filters.mua = [300, 5000]; % MUA filter range (Hz)
    config.trigger_channel = 93;  % Default analog trigger channel
    config.channels = [];             % Auto-detect raw channels
    config.trigger_method = 'analog'; % 'events' or 'analog'
    config.trigger_threshold = 50;   % Threshold for analog trigger detection
    config.event_entity_id = 1;       % Default Event Entity ID
    config.artifact_threshold = 3;    % Z-score threshold for artifact rejection
    config.checksync = 0;
    config.get_deviant = 0;
end
