%% Sliding window behavioral analysis
% chase m 2025

% Clear workspace
clear; close all; clc;

%% Configuration Section
%/Users/chase/Desktop/NKI/data/AV40/peter/pt056057/AttAudThal/pt056057022_analog_imported.mat
pathName = '/Users/chase/Desktop/NKI/data/AV40/peter/pt056057/'; % Eyelink import looks in here for 'edf' subdir
fileName = 'pt056057021'; % ephys and Eyelink file


% Event type selection (options: 'ADDT', 'VDDT', 'VST'), VST under ...
% construction!
eventType = 'ADDT';

% Sliding window configuration (in seconds)
windowSize = 60; 
stepSize = 5;
minEventsThreshold = 1; % event = std or deviant

% Eyelink sampling rate
eyelinkFs = 1000; 


%% Load Eyelink Data
disp('Loading EyelinkData...');
%load(fullfile(pathName, fileName), 'EyelinkData');
EyelinkData = AV40_importEyelink(pathName, fileName);
% Validate necessary fields
requiredFields = {'ADDT_STANDARD', 'ADDT_DEVIANT', 'VDDT_STANDARD', 'VDDT_DEVIANT', 'VST_ONSET', ...
                  'FALSE_ALARM_BUTTON_RESPONSE', 'DEVIANT_HIT_REWARD_JUICE_ONSET'};

missingFields = setdiff(requiredFields, fieldnames(EyelinkData));
if ~isempty(missingFields)
    error('Missing required fields: %s', strjoin(missingFields, ', '));
end


%% Convert Event Times to Seconds

ADDT_STANDARD = EyelinkData.ADDT_STANDARD / eyelinkFs;
ADDT_DEVIANT = EyelinkData.ADDT_DEVIANT / eyelinkFs;
VDDT_STANDARD = EyelinkData.VDDT_STANDARD / eyelinkFs;
VDDT_DEVIANT = EyelinkData.VDDT_DEVIANT / eyelinkFs;
VST_ONSET = EyelinkData.VST_ONSET / eyelinkFs;

buttonPressTimes = unique(EyelinkData.FALSE_ALARM_BUTTON_RESPONSE / eyelinkFs);
hitTimes = unique(EyelinkData.DEVIANT_HIT_REWARD_JUICE_ONSET / eyelinkFs);





%% Select Event Type

switch eventType
    case 'ADDT'
        standardTimes = ADDT_STANDARD;
        deviantTimes = ADDT_DEVIANT;
    case 'VDDT'
        standardTimes = VDDT_STANDARD;
        deviantTimes = VDDT_DEVIANT;
    case 'VST' % UNDER CONSTRUCTION
        standardTimes = VST_ONSET;
        deviantTimes = [];
    otherwise
        error('Invalid eventType selected. Choose ADDT, VDDT, or VST.');
end

%% Define Time Windows for Analysis

standardTimes = standardTimes(:);
deviantTimes = deviantTimes(:);

if isempty(deviantTimes)
    combinedTimes = standardTimes;
else
    combinedTimes = [standardTimes; deviantTimes];
end

startTime = min(combinedTimes);
endTime = max(combinedTimes);
%endTime = combinedTimes(307);


windowEdges = startTime:stepSize:(endTime - windowSize);

% This can happen if the window is too big or file is too small 
if isempty(windowEdges)
    disp('not enough data for window size')
end

% Initialize arrays for hit rate, false alarm rate, d-prime, alpha amplitude, delta ITC
hitRate = NaN(size(windowEdges));
falseAlarmRate = NaN(size(windowEdges));
dprime = NaN(size(windowEdges));




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
        hitRate(i) = NaN;
        dprime(i) = NaN;
        falseAlarmRate(i) = NaN;
        
    else
    
    % Calculate rates
    hitRate(i) = length(windowHits) / length(windowDeviants);
    falseAlarmRate(i) = length(windowButtonPresses) / length(windowStandards);
    % Adjust extreme values to avoid infinite z-scores with dprime
    if hitRate(i) == 1, hitRate(i) = 0.99; end
    if hitRate(i) == 0, hitRate(i) = 0.01; end
    if falseAlarmRate(i) == 1, falseAlarmRate(i) = 0.99; end
    if falseAlarmRate(i) == 0, falseAlarmRate(i) = 0.01; end
    
    dprime(i) = norminv(hitRate(i)) - norminv(falseAlarmRate(i));
    end

  





end

fig1 = figure;

% **1. Hit Rate**
subplot(3,1,1);
plot(windowEdges, hitRate, '-o', 'LineWidth', 2);
title('Hit Rate Over Time');
xlabel('Time (s)');
ylabel('Hit Rate');
ylim([0 1]);
grid on;

% **2. False Alarm Rate**
subplot(3,1,2);
plot(windowEdges, falseAlarmRate, '-o', 'LineWidth', 2);
title('False Alarm Rate Over Time');
xlabel('Time (s)');
ylabel('False Alarm Rate');
ylim([0 1]);
grid on;

% **3. d-prime Sensitivity**
subplot(3,1,3);
plot(windowEdges, dprime, '-o', 'LineWidth', 2, 'Color', 'k');
 sig = ones(1,length(windowEdges));
 hold on
plot(windowEdges,sig)
title('d'' (Sensitivity) Over Time');
xlabel('Time (s)');
ylabel('d'' Value');
grid on;





function timingResults = AV40_importEyelink(dataDirectory, filename)
    % AV40_IMPORT_EYELINK - Import and parse TIMING_RESULTS and DEVIANT_RESPONSE_RESULTS
    %
    % Inputs:
    %   dataDirectory: Path to the main data dir containing edf subdir
    %   filename: Name of the current data file (e.g., 'pt027000029.nev')
    %
    % Output:
    %   timingResults: Struct containing extracted event times for specified events
    
    %% Step 1: Locate the 'edf' Subdirectory
    edfDir = fullfile(dataDirectory, 'edf');
    if ~isfolder(edfDir)
        error('The ''edf'' subdirectory does not exist in: %s', dataDirectory);
    end
    
    % Remove file extension
    [~, baseFilename, ~] = fileparts(filename);
    %fprintf('Searching for subdirectory: %s in %s\n', baseFilename, edfDir);

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

    % Extract headers
    headers = strsplit(rawLines{1}, '\t');
    eventTimeIdx = find(strcmp(headers, 'EVENT_TIME'));

    if isempty(eventTimeIdx)
        warning('EVENT_TIME column not found in TIMING_RESULTS.txt');
    end

    % Process data lines
    for i = 2:length(rawLines)  % Start from second line (skip headers)
        line = strtrim(rawLines{i});
        if contains(line, 'ADDT_STANDARD_ONSET')
            timingResults.ADDT_STANDARD(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'ADDT_DEVIANT_ONSET')
            timingResults.ADDT_DEVIANT(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'VDDT_STANDARD_ONSET')
            timingResults.VDDT_STANDARD(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'VDDT_DEVIANT_ONSET')
            timingResults.VDDT_DEVIANT(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'VST_ONSET')
            timingResults.VST_ONSET(end + 1) = extract_event_time(line, eventTimeIdx);
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

        % Extract headers
        headers = strsplit(rawLines{1}, '\t');
        eventTimeIdx = find(strcmp(headers, 'EVENT_TIME'));

        if isempty(eventTimeIdx)
            warning('EVENT_TIME column not found in DEVIANT_RESPONSE_RESULTS.txt');
        end

        % Process data lines
        for i = 2:length(rawLines)
            line = strtrim(rawLines{i});
            if contains(line, 'DEVIANT_HIT_REWARD_JUICE_ONSET')
                timingResults.DEVIANT_HIT_REWARD_JUICE_ONSET(end + 1) = extract_event_time(line, eventTimeIdx);
            elseif contains(line, 'FALSE_ALARM_BUTTON_RESPONSE')
                timingResults.FALSE_ALARM_BUTTON_RESPONSE(end + 1) = extract_event_time(line, eventTimeIdx);
            elseif contains(line, 'DEVIANT_MISS')
                timingResults.DEVIANT_MISS(end + 1) = extract_event_time(line, eventTimeIdx);
            end
        end
    else
        warning('Skipping DEVIANT_RESPONSE_RESULTS parsing (file missing).');
    end

    %% Helper Function to Extract Event Time
    function eventTime = extract_event_time(line, eventTimeIdx)
        % Extract EVENT_TIME using the correct column index
        tokens = strsplit(line, '\t');
        eventTime = NaN;

        if isempty(eventTimeIdx)
            warning('EVENT_TIME column not detected, returning NaN.');
            return;
        end
        
        if length(tokens) >= eventTimeIdx
            eventTime = str2double(tokens{eventTimeIdx});
            if isnan(eventTime)
                warning('Failed to extract valid EVENT_TIME from line: %s', line);
            end
        else
            warning('EVENT_TIME column index exceeds available columns in line: %s', line);
        end
    end
end