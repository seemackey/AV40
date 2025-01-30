%% Eyelink Timing & Wavelet Analysis Script
% Chase M, 2025

% Clear workspace
clear; close all; clc;

%% Configuration Section
pathName = '/Volumes/Samsung03/data/AV40/Peter/pt030/imported/';
fileName = 'pt030000016.nev_imported.mat'; % Eyelink file

% Event type selection (options: 'ADDT', 'VDDT', 'VST')
eventType = 'ADDT';

% Sliding window configuration (in seconds)
windowSize = 240; 
stepSize = 5;
minEventsThreshold = 5; 

% Eyelink sampling rate
eyelinkFs = 1000; 

% Wavelet parameters
fs = 1000; % Sampling rate
freqRange = [1, 100]; % Broad frequency range
alphaBand = [8, 14]; % Alpha amplitude
deltaFreq = 1.6; % Delta for ITC
preStimulusWindow = -300:-100; % Pre-stimulus period in ms

%% Load Eyelink Data
disp('Loading EyelinkData...');
load(fullfile(pathName, fileName), 'EyelinkData');

% Validate necessary fields
requiredFields = {'ADDT_STANDARD', 'ADDT_DEVIANT', 'VDDT_STANDARD', 'VDDT_DEVIANT', 'VST_ONSET', ...
                  'FALSE_ALARM_BUTTON_RESPONSE', 'DEVIANT_HIT_REWARD_JUICE_ONSET'};

missingFields = setdiff(requiredFields, fieldnames(EyelinkData));
if ~isempty(missingFields)
    error('Missing required fields: %s', strjoin(missingFields, ', '));
end

%% Convert Event Times to Seconds
disp('Converting Eyelink event times to seconds...');
ADDT_STANDARD = EyelinkData.ADDT_STANDARD / eyelinkFs;
ADDT_DEVIANT = EyelinkData.ADDT_DEVIANT / eyelinkFs;
VDDT_STANDARD = EyelinkData.VDDT_STANDARD / eyelinkFs;
VDDT_DEVIANT = EyelinkData.VDDT_DEVIANT / eyelinkFs;
VST_ONSET = EyelinkData.VST_ONSET / eyelinkFs;

buttonPressTimes = unique(EyelinkData.FALSE_ALARM_BUTTON_RESPONSE / eyelinkFs);
hitTimes = unique(EyelinkData.DEVIANT_HIT_REWARD_JUICE_ONSET / eyelinkFs);

%% Select Event Type
disp('Selecting event type...');
switch eventType
    case 'ADDT'
        standardTimes = ADDT_STANDARD;
        deviantTimes = ADDT_DEVIANT;
    case 'VDDT'
        standardTimes = VDDT_STANDARD;
        deviantTimes = VDDT_DEVIANT;
    case 'VST'
        standardTimes = VST_ONSET;
        deviantTimes = [];
    otherwise
        error('Invalid eventType selected. Choose ADDT, VDDT, or VST.');
end

%% Define Time Windows for Analysis
disp('Defining time windows for analysis...');
standardTimes = standardTimes(:);
deviantTimes = deviantTimes(:);

if isempty(deviantTimes)
    combinedTimes = standardTimes;
else
    combinedTimes = [standardTimes; deviantTimes];
end

startTime = min(combinedTimes);
endTime = max(combinedTimes);
windowEdges = startTime:stepSize:(endTime - windowSize);

% Initialize arrays for hit rate, false alarm rate, d-prime, alpha amplitude, delta ITC
hitRate = NaN(size(windowEdges));
falseAlarmRate = NaN(size(windowEdges));
dprime = NaN(size(windowEdges));


%% Load Continuous Data File
contFileName = strrep(fileName, 'imported.mat', 'continuous.mat');
disp(['Loading continuous data from ', contFileName, '...']);
load(fullfile(pathName, contFileName), 'continuous_raw');
continuousData = continuous_raw;

numChannels = size(continuousData, 1); % Number of channels
numChunks = size(continuousData, 2); % 1-second chunks
chunkLength = size(continuousData, 3); % 1000 timepoints per chunk
alphaAmp = NaN(numChannels, length(windowEdges));
deltaITC = NaN(numChannels, length(windowEdges));
chunkStartTimes = (0:numChunks-1); % 1s chunks starting at 0s, 1s, 2s...

%% Use `matfile` for Efficient Saving
resultsFile = fullfile(pathName, sprintf('%s_PerfAndAlphaDeltaOverTime.mat', fileName));
m = matfile(resultsFile, 'Writable', true);

%% Perform Sliding Window Analysis
disp('Performing sliding window analysis...');

for i = 1:length(windowEdges)
    windowStart = windowEdges(i);
    windowEnd = windowStart + windowSize;

    % Get behavioral events in the current time window
    windowStandards = standardTimes(standardTimes >= windowStart & standardTimes <= windowEnd);
    windowDeviants = deviantTimes(deviantTimes >= windowStart & deviantTimes <= windowEnd);
    windowButtonPresses = buttonPressTimes(buttonPressTimes >= windowStart & buttonPressTimes <= windowEnd);
    windowHits = hitTimes(hitTimes >= windowStart & hitTimes <= windowEnd);

    % Skip calculation if insufficient events
    if length(windowStandards) < minEventsThreshold || length(windowDeviants) < minEventsThreshold
        continue;
    end
    
    % Calculate rates
    hitRate(i) = length(windowHits) / length(windowDeviants);
    falseAlarmRate(i) = length(windowButtonPresses) / length(windowStandards);
    dprime(i) = norminv(hitRate(i)) - norminv(falseAlarmRate(i));

    % Find 1s chunks in the window
    chunkIdx = find(chunkStartTimes >= windowStart & chunkStartTimes < windowEnd);
    if isempty(chunkIdx)
        warning('No data found for window %d, skipping wavelet analysis.', i);
        continue;
    end

    % **Stitch Data Together**
    stitchedData = [];
    for j = 1:length(chunkIdx)
        stitchedData = [stitchedData, squeeze(continuousData(:, chunkIdx(j), :))];
    end

    % **Compute Wavelet Transform on Full Window**
    waveletAmp = cell(1, numChannels);
    waveletPhase = cell(1, numChannels);
    
    for ch = 1:numChannels
        [ampFull, phaseFull] = wvlt_bndlm_fxn(stitchedData(ch, :), fs, freqRange, alphaBand);
        [~, phaseDeltaFull] = wvlt_bndlm_fxn(stitchedData(ch, :), fs, freqRange, [deltaFreq, deltaFreq]);

        waveletAmp{ch} = ampFull;
        waveletPhase{ch} = phaseDeltaFull;
    end

    % **Extract Pre-Stimulus Data from Wavelet Output**
    % Compute pre-stimulus indices for all standard stimuli
    preStimIdx = round(preStimulusWindow * (fs / 1000)) + (round(windowStandards * fs) - round(windowStart * fs))';

    % Flatten the index array
    preStimIdx = preStimIdx(:);

    % Remove out-of-bounds indices
    preStimIdx = preStimIdx(preStimIdx > 0 & preStimIdx <= length(stitchedData));

    if isempty(preStimIdx)
        continue;
    end

    for ch = 1:numChannels
        preStimAlphaAmp = waveletAmp{ch}(preStimIdx);
        preStimDeltaPhase = waveletPhase{ch}(preStimIdx);

        % **Compute Mean Alpha Amplitude**
        alphaAmp(ch, i) = mean(preStimAlphaAmp);

        % **Compute Delta ITC**
        deltaITC(ch, i) = abs(mean(exp(1i * preStimDeltaPhase)));
    end

    % Save incrementally
    m.alphaAmp = alphaAmp;
    m.deltaITC = deltaITC;
    m.windowEdges = windowEdges;
    m.hitRate = hitRate;
    m.falseAlarmRate = falseAlarmRate;
    m.dprime = dprime;
end




%% Save Results
bandLabels = strjoin(arrayfun(@(b) sprintf('%d-%dHz', targetBands{b}(1), targetBands{b}(2)), 1:numBands, 'UniformOutput', false), '_');
resultsFile = fullfile(pathName, sprintf('%s_PerfAnd%sOverTime.mat', fileName, bandLabels));

save(resultsFile, 'windowEdges', 'hitRate', 'falseAlarmRate', 'dprime', ...
     'waveletAmplitudes', 'waveletPhases', 'windowSize', 'stepSize', 'eventType', 'targetBands');

disp(['Results saved to ', resultsFile]);
