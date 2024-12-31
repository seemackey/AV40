% import ephys from the ripple system
% chase m 2024


% Clear workspace and initialize
clear;
close all;
clc;

% Directories and file names
    directory1  = '/Volumes/Samsung03/data/AV40/Peter/';
    directory2  = '/Volumes/Samsung03/data/AV40/Peter/imported/';
inputDir = '/Volumes/Samsung03/data/AV40/Peter'; % Replace with your input directory
outputDir = '/Volumes/Samsung03/data/AV40/Peter/imported'; % Replace with your output directory
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

try
    % Call the main data import function
    data_import_v2(inputDir, fileName, config);
    fprintf('importing\n');
catch ME
    % Handle errors 
    fprintf('An error occurred: %s\n', ME.message);
end

function data_import_v2(directory1, fileName, config)
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
        triggers = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold);
    else %digital triggers don't work yet, sorrrryyyyyy
        disp('no analog triggers? check analog trig chan')
        triggers = get_event_triggers(hFile, config.event_entity_id);
    end

    % get Data
    rawData = get_channel_data(hFile, config.channels);
    
    

    % Check if remapping is needed
    if isfield(config, 'channel_remap') == 1
        rawData = remap_channels(rawData);
    end

    


    % Filter Raw Data
    filtertype=1;
    [~, cnte, cntm, cntc, ~, ~] = module_cnt05(rawData, config.newadrate, config.filters.lfp, config.filters.mua, filtertype);

    % Epoch Data
    [eegLFP, eegMUA, eegCSD] = epoch_data(cnte, cntm, cntc, triggers, config);

    % Reject Artifacts
    [eegLFP, eegMUA, eegCSD, triggers] = reject_artifacts(eegLFP, eegMUA, eegCSD, triggers, config);
    
    % get the eyelink data
    timingResults = AV40_importEyelink(directory1, fileName);

    % Save Results
    save_results(directory1, fileName, eegLFP, eegMUA, eegCSD, triggers,timingResults);

    
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





function triggers = get_analog_triggers(hFile, analogChannel, threshold)
    % GET_ANALOG_TRIGGERS - Detect first pulse of each pulse train in analog data
    %
    % Parameters:
    % hFile          - Handle to the Neuroshare file
    % analogChannel  - Analog channel ID
    % threshold      - Threshold for detecting rising edges
    % fs             - Sampling rate (Hz) of the analog data
    %
    % Output:
    % triggers       - Indices of the first pulse of each train
    
    % Retrieve analog data
    [ns_RESULT, ~, analogData] = ns_GetAnalogData(hFile, analogChannel, 1, hFile.Entity(analogChannel).Count);

    % Error handling for invalid data retrieval
    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Error retrieving analog data: %s', ns_RESULT);
    end
    
    % Given string
    samplingRateStr = '30 ksamp/sec';

    % Check if the string matches '30 ksamp/sec'
    if strcmp(samplingRateStr, '30 ksamp/sec')
        fs = 30000; % highest fs in ripple system
        disp('found 30 ksamp/sec.');
    else
        disp('maybe not 30 ksamp/sec.');
    end
    % Detect all rising edges where the signal crosses the threshold
    allTriggers = find(diff(analogData > threshold) == 1); % Rising edge detection
    
    % Time difference threshold to separate trains
    % For 40 Hz pulse trains, period = 1/40 = 0.025 sec (25 ms)
    % For 100 Hz pulse trains, period = 1/100 = 0.01 sec (10 ms)
    minPulseGap = 0.15 * fs; % Minimum gap in samples to consider a new pulse train

    % Initialize the list of first pulses
    triggers = [];
    lastPulse = -inf; % Track the last pulse's index

    % Loop through all detected triggers and just get stim onset
    for i = 1:length(allTriggers)
        if allTriggers(i) - lastPulse > minPulseGap
            % If enough time has passed, this is the start of a new train
            triggers(end + 1) = allTriggers(i); %#ok<AGROW>
            lastPulse = allTriggers(i);
        end
    end

    % Optional: Plot the signal and detected triggers
%     figure;
%     plot(1:length(analogData), analogData);
%     hold on;
%     plot(triggers, analogData(triggers), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
%     title('Analog Signal with Detected Triggers');
%     xlabel('Sample Index');
%     ylabel('Analog Signal');
%     legend('Analog Signal', 'First Pulse of Train');
%     hold off;
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

function save_results(directory, fileName, eegLFP, eegMUA, eegCSD, analogData, triggers,eyelinkTiming)
    % Save processed data
    save(fullfile(directory, [fileName '_epoched.mat']), 'eegLFP', 'eegMUA', 'eegCSD', 'triggers','eyelinkTiming');
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
end
