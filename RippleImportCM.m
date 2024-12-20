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
config.channels = [2:25];             % ephys data channels
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
    else
        triggers = get_event_triggers(hFile, config.event_entity_id);
    end

    % get Raw Data
    config.channels;
    rawData = get_channel_data(hFile, rawChannels);
    
    % get the eyelink data
    timingResults = import_timing_results(directory1, filename);

    % Filter Raw Data
    [lfp, mua, csd] = filter_data(rawData, config);

    % Epoch Data
    [eegLFP, eegMUA, eegCSD] = epoch_data(lfp, mua, csd, triggers, config);

    % Reject Artifacts
    [eegLFP, eegMUA, eegCSD, triggers] = reject_artifacts(eegLFP, eegMUA, eegCSD, triggers, config);

    % Save Results
    save_results(directory1, fileName, eegLFP, eegMUA, eegCSD, analogData, triggers);

    
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
    % Retrieve data for specified channels
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

    % Loop through all detected triggers
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



function timingResults = import_timing_results(dataDirectory, filename)
    % IMPORT_TIMING_RESULTS - Imports the TIMING_RESULTS file
    % Searches for a subdirectory in 'edf' matching the filename (minus extension),
    % then imports 'TIMING_RESULTS' from within the matching subdirectory.
    %
    % Parameters:
    % dataDirectory: The main data directory containing the edf folder.
    % filename: Name of the current data file (e.g., 'pt027000029.nev').
    %
    % Output:
    % timingResults: Struct containing extracted timing data.

    %% Step 1: Locate the 'edf' Subdirectory
    edfDir = fullfile(dataDirectory, 'edf');
    if ~isfolder(edfDir)
        error('The ''edf'' subdirectory does not exist in: %s', dataDirectory);
    end

    % Remove file extension from filename
    [~, baseFilename, ~] = fileparts(filename);
    fprintf('Searching for subdirectory: %s in %s\n', baseFilename, edfDir);

    %% Step 2: Search for Matching Subdirectory
    subdirs = dir(edfDir);  % List all items in the 'edf' directory
    isDir = [subdirs.isdir];
    subdirs = subdirs(isDir);  % Keep only directories

    % Find a subdirectory that matches the filename
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

    %% Step 3: Locate and Import TIMING_RESULTS
    timingFilePath = fullfile(matchingSubdir, 'TIMING_RESULTS');
    if ~exist(timingFilePath, 'file')
        error('TIMING_RESULTS file not found in: %s', matchingSubdir);
    end
    
    fprintf('Loading TIMING_RESULTS from: %s\n', timingFilePath);
    [fid, message] = fopen(timingFilePath, 'rt');
    if fid == -1
        error('Error opening TIMING_RESULTS file: %s', message);
    end
    
    % Initialize storage for lines
    timingResults.RawLines = {};
    I = 1;

    % Read file line by line
    while ~feof(fid)
        timingResults.RawLines{I, 1} = fgetl(fid);
        I = I + 1;
    end
    fclose(fid);
    
    %% Step 4: Extract Relevant Timing Information
    fprintf('Extracting timing information...\n');
    timingResults.nLines = length(timingResults.RawLines);
    timingResults.TrialStartTimes = [];
    
    for lineIdx = 1:timingResults.nLines
        currentLine = strtrim(timingResults.RawLines{lineIdx});
        
        if contains(currentLine, 'TRIALID')
            splitLine = strsplit(currentLine);
            trialTime = str2double(splitLine{2});
            if ~isnan(trialTime)
                timingResults.TrialStartTimes(end + 1) = trialTime; %#ok<AGROW>
            end
        end
    end
    
    %% Step 5: Summary
    fprintf('Total lines read: %d\n', timingResults.nLines);
    fprintf('Trial start times extracted: %d\n', length(timingResults.TrialStartTimes));
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

function save_results(directory, fileName, eegLFP, eegMUA, eegCSD, analogData, triggers)
    % Save processed data
    save(fullfile(directory, [fileName '_processed.mat']), 'eegLFP', 'eegMUA', 'eegCSD', 'analogData', 'triggers');
end

function config = get_default_config()
    % Define default processing parameters
    config.epoch_tframe = [-50, 150]; % Epoch window in ms
    config.newadrate = 1000;          % Resampling rate
    config.filters.lfp = [0.5, 300];  % LFP filter range (Hz)
    config.filters.mua = [300, 5000]; % MUA filter range (Hz)
    config.trigger_channel = 10241;  % Default analog trigger channel
    config.channels = [];             % Auto-detect raw channels
    config.trigger_method = 'analog'; % 'events' or 'analog'
    config.trigger_threshold = 0.5;   % Threshold for analog trigger detection
    config.event_entity_id = 1;       % Default Event Entity ID
    config.artifact_threshold = 3;    % Z-score threshold for artifact rejection
end
