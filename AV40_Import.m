% import ephys from the ripple system
% chase m 2024


% Clear workspace and initialize
clear;
close all;
clc;

% Directories and file names
inputDir = '/Volumes/Samsung03/data/AV40/Peter'; % Replace with your input directory
figuresDir = '/Volumes/Samsung03/data/AV40/Peter/imported'; % Replace with your output directory
fileName = 'pt027000029.nev'; % Replace with your file name

% Example configuration
config = struct();
config.epoch_tframe = [-250, 250]; % Epoch window in ms
config.newadrate = 1000;          % Resampling rate
config.filters.lfp = [0.5, 300];  % LFP filter range (Hz)
config.filters.mua = [300, 5000]; % MUA filter range (Hz)
config.trigger_channel = 93;  %  analog trigger channel for aud
config.channels = [2:25];             % ephys data channels
config.channel_remap = true; % Enable channel remapping by default
config.trigger_method = 'analog'; % 'events' or 'analog'
config.trigger_threshold = 50;   % Threshold for analog trigger detection
config.event_entity_id = 1;       % Default Event Entity ID
config.artifact_threshold = 3;    % Z-score threshold for artifact rejection
config.checksync = 0; % check sync between ripple and eyelink
config.get_deviant = 1; 
config.store_cont_data = 1;

try
    % Call the main data import function
    if ~exist(figuresDir, 'dir')
     mkdir(figuresDir);
    end
    [epoched_data] = data_import_v2(inputDir,figuresDir, fileName, config);
    
    
    
    %% make figs

    plot_baseline_corrected_data(epoched_data.standard, config, 'standard', figuresDir, fileName);

    % Plot deviant
    if isfield(epoched_data, 'deviant') && ~isempty(epoched_data.deviant)
        plot_baseline_corrected_data(epoched_data.deviant, config, 'deviant', figuresDir, fileName);
    end
    
catch ME
    % Handle errors 
    fprintf('An error occurred: %s\n', ME.message);
end

%% FUNCTIONS BELOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [epoched_data] = data_import_v2(directory1, figuresDir, fileName, config)
    % DATA_IMPORT_V2 - Trigger-based Import Script
    
    %% Open File
    filePath = fullfile(directory1, fileName);
    [ns_RESULT, hFile] = ns_OpenFile(filePath);
    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Failed to open file: %s', ns_RESULT);
    end
    
    %% Get Trigger Times
    triggerChannel = config.trigger_channel;
    [triggers_std, triggers_deviant] = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold);
    % get the eyelink data
    EyelinkData = AV40_importEyelink(directory1, fileName);
    %% Sampling Rates and Epoch Setup
    fs = 30000; % Ripple sampling rate (Hz)
    newadrate = config.newadrate;
    x1 = round(config.epoch_tframe(1) * (fs / 1000));
    x2 = round(config.epoch_tframe(2) * (fs / 1000));
    epochLength = abs(x1) + abs(x2) + 1; % 30 khz epochs for retrieval
    
    %% Pre-allocate Matrices
    numChannels = length(config.channels);
    numStdTriggers = length(triggers_std);
    numDevTriggers = length(triggers_deviant);
    % Original epoch length (samples at 30 kHz)
    epochLength_orig = abs(x1) + abs(x2) + 1; 

    % Downsampled epoch length (samples at 1 kHz)
    epochLength_ds = round(epochLength_orig * (config.newadrate / fs)); 

    
    % Epoch matrices need to account for the trigger sample (+1)
    lfp_std = NaN(numChannels, numStdTriggers, epochLength_ds+1);
    mua_std = NaN(numChannels, numStdTriggers, epochLength_ds+1);
    lfp_dev = NaN(numChannels, numDevTriggers, epochLength_ds+1);
    mua_dev = NaN(numChannels, numDevTriggers, epochLength_ds+1);
    
    %% Epoch Data for Standards
    disp('Importing triggered data for standards...');
    for chIdx = 1:numChannels
        ch = config.channels(chIdx);
        for tIdx = 1:numStdTriggers
            trigger = triggers_std(tIdx);
            startIndex = trigger + x1;
            endIndex = trigger + x2;

            if startIndex < 1 || endIndex > hFile.Entity(ch).Count
                warning('Invalid indices for trigger %d on channel %d. Skipping...', tIdx, ch);
                continue;
            end

            [ns_RESULT, ~, epochData] = ns_GetAnalogData(hFile, ch, startIndex, epochLength);
            if strcmp(ns_RESULT, 'ns_OK') && numel(epochData) == epochLength
                [lfp, mua] = AV40_filt(epochData, newadrate, config.filters.lfp, config.filters.mua, fs);
                lfp_std(chIdx, tIdx, :) = lfp;
                mua_std(chIdx, tIdx, :) = mua;
            end
        end
    end

    %% Remap Channels (if enabled)
    
        lfp_std = remap_channels(lfp_std);
        mua_std = remap_channels(mua_std);
    
    
    %% Compute CSD and Bipolar Signals
    
    csd_std = -diff(lfp_std, 2, 1);
    lfp_std = lfp_std(2:end-1,:,:);
    mua_std = mua_std(2:end-1,:,:);
    
    %% Epoch Data for Deviants
    disp('Importing triggered data for deviants...');
    for chIdx = 1:numChannels
        ch = config.channels(chIdx);
        for tIdx = 1:numDevTriggers
            trigger = triggers_deviant(tIdx);
            startIndex = trigger + x1;
            endIndex = trigger + x2;

            if startIndex < 1 || endIndex > hFile.Entity(ch).Count
                warning('Invalid indices for trigger %d on channel %d. Skipping...', tIdx, ch);
                continue;
            end

            [ns_RESULT, ~, epochData] = ns_GetAnalogData(hFile, ch, startIndex, epochLength);
            if strcmp(ns_RESULT, 'ns_OK') && numel(epochData) == epochLength
                [lfp, mua] = AV40_filt(epochData, newadrate, config.filters.lfp, config.filters.mua, fs);
                lfp_dev(chIdx, tIdx, :) = lfp;
                mua_dev(chIdx, tIdx, :) = mua;
            end
        end
    end
    
    %% Remap and Compute CSD for Deviants
    if config.channel_remap
        lfp_dev = remap_channels(lfp_dev);
        mua_dev = remap_channels(mua_dev);
    end
    
    csd_dev = -diff(lfp_dev, 2, 1);
    lfp_dev = lfp_dev(2:end-1,:,:);
    mua_dev = mua_dev(2:end-1,:,:);
    
    %% Import Continuous Data in 1-Second Chunks (Downsampled to 1 kHz)
    if config.store_cont_data == 1
    disp('Importing and downsampling continuous raw data into 1-second chunks...');

    chunkSize = fs; % 1-second chunk at original sampling rate (30 kHz)
    chunkSize_ds = config.newadrate; % Downsampled size (1 kHz per second)
    totalSamples = hFile.Entity(config.channels(1)).Count; % Total samples in each channel

    numChunks = ceil(totalSamples / chunkSize);

    % Pre-allocate downsampled continuous data
    continuous_raw = NaN(numChannels, numChunks, chunkSize_ds);
    read_me_plz = 'chunk size for cont data is 1 second';
    for chIdx = 1:numChannels
        ch = config.channels(chIdx);
        for chunkIdx = 1:numChunks
            % Define chunk start and end indices
            startSample = (chunkIdx - 1) * chunkSize + 1;
            endSample = min(chunkIdx * chunkSize, totalSamples);
            chunkLength = endSample - startSample + 1;

            % Ensure valid chunk range
            if startSample <= totalSamples
                try
                    % Retrieve raw data chunk
                    [ns_RESULT, ~, chunkData] = ns_GetAnalogData(hFile, ch, startSample, chunkLength);
                    if strcmp(ns_RESULT, 'ns_OK') && ~isempty(chunkData)
                        % Downsample the chunk
                        chunkData_ds = resample(chunkData, chunkSize_ds, chunkLength);
                        continuous_raw(chIdx, chunkIdx, 1:length(chunkData_ds)) = chunkData_ds';
                    else
                        warning('Failed to retrieve or downsample chunk %d on channel %d.', chunkIdx, ch);
                    end
                catch ME
                    warning('Error processing chunk %d on channel %d: %s', chunkIdx, ch, ME.message);
                end
            else
                warning('Chunk %d on channel %d out of bounds. Skipping...', chunkIdx, ch);
            end
        end
    end

    % Save Continuous Raw Data
    
    disp('Continuous raw data imported and downsampled successfully.');
    else
        continuous_raw = [];
        read_me_plz = 'continuous data was not stored';
    end

    
    %% Save Data
    epoched_data.standard.lfp = lfp_std;
    epoched_data.deviant.lfp = lfp_dev;
    epoched_data.standard.csd = csd_std;
    epoched_data.deviant.csd = csd_dev;
    epoched_data.standard.mua = mua_std;
    epoched_data.deviant.mua = mua_dev;
    
    save(fullfile(figuresDir, [fileName '_imported.mat']), 'epoched_data', 'triggers_std', 'triggers_deviant','EyelinkData', 'config','continuous_raw','read_me_plz');
    disp('Data import complete!');
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
    if config.epoch_tframe(1) < -50
       baselineStart = -50; % Start of baseline window (ms)
       baselineEnd = -5; % End of baseline window (ms)
    else
        baselineStart = config.epoch_tframe(1); % Start of baseline window (ms)
        baselineEnd = -5; % End of baseline window (ms)
    end
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
    [~,samplingRateStr] = hFile.FileInfo.Label;
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
