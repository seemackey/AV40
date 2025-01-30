%% Eyelink Timing Data Analysis Script (With d-prime Calculation)
% Chase M, 2025

% Clear workspace and initialize
clear; close all; clc;

% Configuration Section
pathName = '/Volumes/Samsung03/data/AV40/Peter/pt030/imported/'; % <-- Set the path to your imported data
fileName = 'pt030000016.nev_imported.mat'; % <-- Set the filename

% Event type selection (options: 'ADDT', 'VDDT', 'VST')
eventType = 'ADDT';

% Sliding window configuration (in seconds)
windowSize = 240; % Size of the analysis window
stepSize = 5;     % Step size for moving the window
minEventsThreshold = 5; % Minimum standards or deviants per window


% Eyelink sampling rate (in Hz)
eyelinkFs = 1000; % EyelinkData sampled at 1kHz

% Load Imported Data File
disp('Loading EyelinkData from imported file...');
load(fullfile(pathName, fileName), 'EyelinkData');

% Validate necessary fields
requiredFields = {'ADDT_STANDARD', 'ADDT_DEVIANT', 'VDDT_STANDARD', 'VDDT_DEVIANT', 'VST_ONSET', ...
                  'FALSE_ALARM_BUTTON_RESPONSE', 'DEVIANT_HIT_REWARD_JUICE_ONSET'};

missingFields = setdiff(requiredFields, fieldnames(EyelinkData));
if ~isempty(missingFields)
    error('Missing required fields: %s', strjoin(missingFields, ', '));
end

% Convert Event Times to Seconds
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

%% Define Time Window for Sliding Analysis
disp('Defining time window for sliding analysis...');
standardTimes = standardTimes(:); % Convert to column vector
deviantTimes = deviantTimes(:);   

if isempty(deviantTimes)
    combinedTimes = standardTimes;
else
    combinedTimes = [standardTimes; deviantTimes];
end

startTime = min(combinedTimes);
endTime = max(combinedTimes);
windowEdges = startTime:stepSize:(endTime - windowSize);

% Initialize arrays for hit rate, false alarm rate, and d-prime
hitRate = NaN(size(windowEdges));
falseAlarmRate = NaN(size(windowEdges));
dprime = NaN(size(windowEdges));

%% Sliding Window Analysis
disp('Performing sliding window analysis...');

for i = 1:length(windowEdges)
    windowStart = windowEdges(i);
    windowEnd = windowStart + windowSize;

    % Get events in the current time window
    windowStandards = standardTimes(standardTimes >= windowStart & standardTimes <= windowEnd);
    windowDeviants = deviantTimes(deviantTimes >= windowStart & deviantTimes <= windowEnd);
    windowButtonPresses = buttonPressTimes(buttonPressTimes >= windowStart & buttonPressTimes <= windowEnd);
    windowHits = hitTimes(hitTimes >= windowStart & hitTimes <= windowEnd);

    % Debug print event counts
    fprintf('Window %d: Standards=%d, Deviants=%d, False Alarms=%d, Hits=%d\n', ...
            i, length(windowStandards), length(windowDeviants), length(windowButtonPresses), length(windowHits));
    
    % Skip calculation if insufficient events
    if length(windowStandards) < minEventsThreshold || length(windowDeviants) < minEventsThreshold
        continue;
    end
    
    % Calculate rates
    hitRate(i) = length(windowHits) / length(windowDeviants);
    falseAlarmRate(i) = length(windowButtonPresses) / length(windowStandards);

    % Adjust extreme values to avoid infinite z-scores
    if hitRate(i) == 1, hitRate(i) = 0.99; end
    if hitRate(i) == 0, hitRate(i) = 0.01; end
    if falseAlarmRate(i) == 1, falseAlarmRate(i) = 0.99; end
    if falseAlarmRate(i) == 0, falseAlarmRate(i) = 0.01; end

    % Compute d'
    dprime(i) = norminv(hitRate(i)) - norminv(falseAlarmRate(i));
end

%% Plot Results
disp('Plotting results...');
figure;

subplot(3,1,1);
plot(windowEdges, hitRate, '-o', 'LineWidth', 2);
title('Hit Rate Over Time');
xlabel('Time (s)');
ylabel('Hit Rate');
ylim([0 1]);
grid on;

subplot(3,1,2);
plot(windowEdges, falseAlarmRate, '-o', 'LineWidth', 2);
title('False Alarm Rate Over Time');
xlabel('Time (s)');
ylabel('False Alarm Rate');
ylim([0 1]);
grid on;

subplot(3,1,3);
plot(windowEdges, dprime, '-o', 'LineWidth', 2, 'Color', 'k');
title('d'' (Sensitivity) Over Time');
xlabel('Time (s)');
ylabel('d'' Value');
grid on;

%% Save Results
% Uncomment to save
resultsFile = fullfile(pathName, [fileName '_PerfOverTime.mat']);
save(resultsFile, 'windowEdges', 'hitRate', 'falseAlarmRate', 'dprime', 'windowSize', 'stepSize', 'eventType');
disp(['Results saved to ', resultsFile]);
