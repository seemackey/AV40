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
config.trigger_channel = 94;  % Default analog trigger channel
config.channels = [2:25];             % Auto-detect raw channels
config.trigger_method = 'analog'; % 'events' or 'analog'
config.trigger_threshold = 0.5;   % Threshold for analog trigger detection
config.event_entity_id = 1;       % Default Event Entity ID
config.artifact_threshold = 3;    % Z-score threshold for artifact rejection

try
    % Call the main data import function
    data_import_v2(inputDir, fileName, config);
    fprintf('Data import and preprocessing complete!\n');
catch ME
    % Handle errors gracefully
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

    % Separate entities into raw and analog categories
    [rawChannels, analogChannels] = classify_channels(hFile);

    % Handle triggers
    if strcmp(config.trigger_method, 'analog')
        triggerChannel = config.trigger_channel; % Specify analog channel for triggers
        triggers = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold);
    else
        triggers = get_event_triggers(hFile, config.event_entity_id);
    end

    % Process Raw Data
    rawData = get_channel_data(hFile, rawChannels);

    % Process Analog Data (e.g., eye position, stimulus)
    analogData = get_channel_data(hFile, analogChannels);

    % Filter Raw Data
    [lfp, mua, csd] = filter_data(rawData, config);

    % Epoch Data
    [eegLFP, eegMUA, eegCSD] = epoch_data(lfp, mua, csd, triggers, config);

    % Reject Artifacts
    [eegLFP, eegMUA, eegCSD, triggers] = reject_artifacts(eegLFP, eegMUA, eegCSD, triggers, config);

    % Save Results
    save_results(directory1, fileName, eegLFP, eegMUA, eegCSD, analogData, triggers);

    % Close File
    ns_CloseFile(hFile);
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
    % Detect triggers in analog data
    
    % Retrieve analog data
    [ns_RESULT, ContCount, analogData] = ns_GetAnalogData(hFile, analogChannel, 1, hFile.Entity(analogChannel).Count);

    % Error handling for invalid data retrieval
    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Error retrieving analog data: %s', ns_RESULT);
    end
    plot(1:length(analogData),analogData)
    % Detect rising edges based on threshold
    triggers = find(diff(analogData > threshold) == 1); % Rising edge detection
end


function [lfp, mua, csd] = filter_data(rawData, config)
    % Filter raw data into LFP, MUA, and CSD
    lfp = bandpass_filter(rawData, config.filters.lfp, config.newadrate);
    mua = bandpass_filter(rawData, config.filters.mua, config.newadrate);
    csd = calculate_csd(lfp);
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
