%% Sliding window behavior and oscillations
% chase m 2025

% Clear workspace
clear; close all; clc;

%% Configuration Section
%/Users/chase/Desktop/NKI/data/AV40/peter/pt056057/AttAudThal/pt056057022_analog_imported.mat
pathName = 'H:\Peter\pt066067\AttAudThal\';
fileName = 'pt066067024_analog_continuous'; % Eyelink file
% Event type selection (options: 'ADDT', 'VDDT', 'VST'), VST under ...
% construction!
eventType = 'ADDT';

% Sliding window configuration (in seconds)
windowSize = 60; 
stepSize = 5;
minEventsThreshold = 1; 

% Eyelink sampling rate
eyelinkFs = 1000; 
rippleFs = 30000; % assumed Fs of the analog trigs for alignment

% Wavelet parameters
fs = 1000; % Sampling rate for alignment to ephys AND for wavelet
freqRange = [0.5, 36]; % Broad frequency range
alphaBand = [8, 14]; % Alpha amplitude
GP_band = [5 50]; % generalized phase

if contains(eventType,'VDDT')
    deltaFreq = [1,1.3]; % Delta for ITC
else
    deltaFreq = [1.5,1.8];
end

preStimulusWindow = -400:-200; % Pre-stimulus period in ms

% use CSD or bip LFP? (boolean)
csd = 1;
selchan = 6;

selchansforcorrelation = 1:22; % Spearman Corr figure

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

%% Load Continuous Data File
contFileName = strrep(fileName, 'imported.mat', 'continuous.mat');
%contFileName = 'pt033000018.nev_continuous.mat';
%contFileName = 'pt031032019.nev_continuous.mat';
disp(['Loading continuous data from ', contFileName, '...']);
load(fullfile(pathName, contFileName));
continuousData = continuous_raw;
%% Convert Event Times to Seconds
disp('Converting Eyelink event times to seconds...');
ADDT_STANDARD = EyelinkData.ADDT_STANDARD / eyelinkFs;
ADDT_DEVIANT = EyelinkData.ADDT_DEVIANT / eyelinkFs;
VDDT_STANDARD = EyelinkData.VDDT_STANDARD / eyelinkFs;
VDDT_DEVIANT = EyelinkData.VDDT_DEVIANT / eyelinkFs;
VST_ONSET = EyelinkData.VST_ONSET / eyelinkFs;

buttonPressTimes = unique(EyelinkData.FALSE_ALARM_BUTTON_RESPONSE / eyelinkFs);
hitTimes = unique(EyelinkData.DEVIANT_HIT_REWARD_JUICE_ONSET / eyelinkFs);

disp('Adjusting time alignment...');
firstEyelinkStandard = min(ADDT_STANDARD); % auditory was always used to sync
firstNeuralStandard = min(triggers_std_analog)/rippleFs; % Convert to seconds

% Compute time offset between datasets
timeOffset = firstNeuralStandard - firstEyelinkStandard;
fprintf('Time Offset Between Neural and Eyelink Data: %.3f sec\n', timeOffset);

% Adjust Eyelink timestamps
ADDT_STANDARD = triggers_std_analog/rippleFs;
ADDT_DEVIANT = ADDT_DEVIANT + timeOffset;
VDDT_STANDARD = VDDT_STANDARD + timeOffset;
VDDT_DEVIANT = VDDT_DEVIANT + timeOffset;
VST_ONSET = VST_ONSET + timeOffset;

buttonPressTimes = buttonPressTimes + timeOffset;
hitTimes = hitTimes + timeOffset;

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



%% preallocate arrays for neural analysis
numChannels = size(continuousData, 1); % Number of channels
numChunks = size(continuousData, 2); % 1-second chunks
chunkLength = size(continuousData, 3); % 1000 timepoints per chunk

chunkStartTimes = (0:numChunks-1); % 1s chunks starting at 0s, 1s, 2s...
    % preallocate
        alphaAmp = NaN(numChannels, length(windowEdges));
deltaITC = NaN(numChannels, length(windowEdges));
gpITC = NaN(numChannels, length(windowEdges));

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
    
    for ch = 1:numChannels
        [stitchedDataFilt(ch,:), ~] = AV40_filt(stitchedData(ch,:), 1000, [0.5 300], [300 400], 1000);
    end
    
    % Derivationo of local signal (e.g. CSD)
        if ~csd==0
            stitchedDataFiltDeriv = -diff(stitchedData,2,1); %Current Source Density
        else
            stitchedDataFiltDeriv = diff(stitchedData,1,1);%Bipolar Field Potential
        end
    % update based on derivation
    numChannelsDeriv = size(stitchedDataFiltDeriv, 1); % Number of channels
    % **Compute Wavelet Transform on Full Window**
    waveletAmp = cell(1, numChannelsDeriv);
    waveletPhase = cell(1, numChannelsDeriv);
    gp_phase = cell(1,numChannelsDeriv);
    
    for ch = 1:numChannelsDeriv

        x = stitchedDataFiltDeriv(ch,:);
        [ampFull, phaseFull] = wvlt_bndlm_fxn(stitchedDataFiltDeriv(ch, :), fs, freqRange, alphaBand);
        [~, phaseDeltaFull] = wvlt_bndlm_fxn(stitchedDataFiltDeriv(ch, :), fs, freqRange, deltaFreq);
        [phi, f_inst] = gp_phase_and_freq_local(x, fs, 0, GP_band);
        waveletAmp{ch} = ampFull;
        waveletPhase{ch} = phaseDeltaFull;
        gp_phase{ch} = phi;
    end

    % **Extract Pre-Stimulus Data from Wavelet Output**
    % Compute pre-stimulus indices for all standard stimuli
    % Compute pre-stimulus indices for all standard stimuli%% Compute Pre-Stimulus Indices for Alpha Amplitude (Full Window)
    preStimIdxFull = bsxfun(@plus, ...
        round(preStimulusWindow * (fs / 1000)), ... % Convert entire pre-stimulus window to sample indices
        round(windowStandards * fs) - round(windowStart * fs)); % Align with window standards

    % Flatten and remove out-of-bounds indices
    preStimIdxFull = preStimIdxFull(:);
    preStimIdxFull = preStimIdxFull(preStimIdxFull > 0 & preStimIdxFull <= length(stitchedDataFiltDeriv));

    % Compute Pre-Stimulus Index for Delta ITC (Single Midpoint)
    preStimMidpointIdx = round(median(preStimulusWindow) * (fs / 1000)); % Convert midpoint to sample index
    preStimIdxMid = round(windowStandards * fs) - round(windowStart * fs) + preStimMidpointIdx;

    % Remove out-of-bounds indices
    preStimIdxMid = preStimIdxMid(preStimIdxMid > 0 & preStimIdxMid <= length(stitchedDataFiltDeriv));

    % Skip iteration if no valid pre-stimulus indices
    if isempty(preStimIdxFull) || isempty(preStimIdxMid)
        continue;
    end


    %% extract Alpha Amplitude and compute Delta ITC
    for ch = 1:numChannelsDeriv
        % **Alpha Amplitude (Mean Over Full Pre-Stim Window)**
        preStimAlphaAmp = waveletAmp{ch}(preStimIdxFull);
        alphaAmp(ch, i) = mean(preStimAlphaAmp);
        alphaStd(ch, i) = std(preStimAlphaAmp);

        % **Delta ITC (Single Time Point at Midpoint)**
        preStimDeltaPhase = waveletPhase{ch}(preStimIdxMid);
        preStimGPphase = gp_phase{ch}(preStimIdxMid);
        deltaITC(ch, i) = abs(mean(exp(1i * preStimDeltaPhase))); % Compute ITC at single midpoint
        gpITC(ch,i) = abs(mean(exp(1i * preStimGPphase))); % Compute ITC at single midpoint
        
    end



end


% Organize results into a structured format
results.alphaAmp = alphaAmp;  % Mean prestimulus alpha amplitude
results.alphaStd = alphaStd;
results.deltaITC = deltaITC;  % Inter-trial coherence (ITC) of prestim delta phase
results.windowEdges = windowEdges; % Time points for analysis
results.hitRate = hitRate;  % Behavioral hit rate
results.falseAlarmRate = falseAlarmRate; % False alarm rate
results.dprime = dprime; % d-prime sensitivity metric
results.eventType = eventType; % Condition (ADDT, VDDT, VST)
results.gpITC = gpITC;

% Include wavelet analysis parameters
results.fs = fs;  % Sampling rate
results.freqRange = freqRange; % Broad wavelet frequency range
results.alphaBand = alphaBand; % Frequency range for alpha amplitude
results.deltaFreq = deltaFreq; % Frequency range for delta ITC
results.GP_band = GP_band;
results.preStimulusWindow = preStimulusWindow; % Pre-stimulus time window in ms


% 
% Remove .nev and .mat extensions from filename
fileBaseName = regexprep(fileName, '\.nev|\.mat', '');

% Create a filename that includes event type, frequency bands, and window size
bandLabels = sprintf('%s_Alpha%d-%dHz_Delta%.1f-%.1fHz_Window%d', ...
    eventType, alphaBand(1), alphaBand(2), deltaFreq(1), deltaFreq(2), windowSize);

resultsFile = fullfile(pathName, sprintf('%s_%sOverTime.mat', fileBaseName, bandLabels));

% Save the struct
save(resultsFile, 'results');

disp(['Results saved to ', resultsFile]);

%% Plot behavior, alpha amp, and delta ITC

disp('Plotting results...');
fig1 = figure;

% **1. Hit Rate**
subplot(5,1,1);
plot(windowEdges, hitRate, '-o', 'LineWidth', 2);
title('Hit Rate Over Time');
xlabel('Time (s)');
ylabel('Hit Rate');
ylim([0 1]);
grid on;

% **2. False Alarm Rate**
subplot(5,1,2);
plot(windowEdges, falseAlarmRate, '-o', 'LineWidth', 2);
title('False Alarm Rate Over Time');
xlabel('Time (s)');
ylabel('False Alarm Rate');
ylim([0 1]);
grid on;

% **3. d-prime Sensitivity**
subplot(5,1,3);
plot(windowEdges, dprime, '-o', 'LineWidth', 2, 'Color', 'k');
title('d'' (Sensitivity) Over Time');
xlabel('Time (s)');
ylabel('d'' Value');
grid on;

% **4. Alpha Amplitude (Prestimulus)**
subplot(5,1,4);
plot(windowEdges, alphaAmp(selchan,:), '-o', 'LineWidth', 2, 'Color', 'b');
title('Prestimulus Alpha Amplitude');
xlabel('Time (s)');
ylabel('Alpha Power');
grid on;

% **5. Delta ITC (Prestimulus)**
subplot(5,1,5);
plot(windowEdges, deltaITC(selchan,:), '-o', 'LineWidth', 2, 'Color', 'r');
hold on
plot(windowEdges, gpITC(selchan,:), '-x', 'LineWidth', 2, 'Color', 'g');
legend('Delta','BB')
title('Prestimulus ITC');
xlabel('Time (s)');
ylabel('Inter-Trial Coherence');
grid on;

% Adjust figure layout
sgtitle(sprintf('Behavioral and Neural Measures Over Time (%s)', eventType));
set(gcf, 'Position', [100, 100, 900, 800]); % Resize figure

fig2 = figure;

% Scatter plot: d' vs. Alpha Amplitude
subplot(1,2,1);
scatter(alphaAmp(selchan,:), dprime, 'b', 'filled'); % 
hold on;
lsline; % Add least-squares regression line
xlabel('Alpha Amplitude');
ylabel('d'' (Sensitivity)');
title('d'' vs. Alpha Amplitude');
grid on;
axis square

% Scatter plot: d' vs. Delta ITC
subplot(1,2,2);
scatter(deltaITC(selchan,:), dprime, 'r', 'filled'); % 
hold on;
lsline; % Add least-squares regression line
xlabel('Delta ITC');
ylabel('d'' (Sensitivity)');
title('d'' vs. Delta ITC');
grid on;
axis square



% Across channel correlations

AV40_PlotLaminarCorrGP(results,selchansforcorrelation);
fig3 = gcf;
% save figures
% Extract directory from resultsFile

[resultsDir, baseFileName, ~] = fileparts(resultsFile);

% Save figure with specific filename
performanceFigFile = fullfile(resultsDir, sprintf('%s_PerfOverTime.jpg', baseFileName));

saveas(fig1, performanceFigFile);
scatters = fullfile(resultsDir, sprintf('%s_Scatter.jpg', baseFileName));
saveas(fig2,scatters);
acrosschanfig = fullfile(resultsDir, sprintf('%s_AcrossChan.jpg', baseFileName));
saveas(fig3,acrosschanfig);


function [phi, f_inst] = gp_phase_and_freq_local(x, Fs, pad_sec, GP_band)
% Reflection-pad -> band-limit -> generalized phase -> instantaneous freq
% Inputs:
%   x        : 1×T (double), single-channel signal
%   Fs       : sampling rate (Hz)
%   pad_sec  : reflection padding in seconds (on both sides)
%   GP_band  : [f1 f2] band (Hz) for GP band-limiting
% Outputs:
%   phi      : 1×T generalized phase (radians)
%   f_inst   : 1×T-1 instantaneous generalized frequency (Hz), NaN-padded to length T

    x = double(x(:));  T = numel(x);
    pad = max(1, round(pad_sec*Fs));

    % reflect padding
    pre  = flipud(x(1:min(pad, T)));
    post = flipud(x(max(1, end-min(pad,T)+1):end));
    xpad = [pre; x; post];
    

    % band-limit
    [b,a] = butter(4, GP_band/(Fs/2));
    xpadf = filtfilt(b,a,xpad);

    % generalized phase
    xgp = generalized_phase_vector(xpadf, Fs, 0);
    phip = angle(xgp(:));

    % slice center (original signal support)
    phi = phip(numel(pre)+(1:T)).';

    % instantaneous generalized frequency (unwrap/diff)
    uphi = unwrap(phi);
    dphi = diff(uphi);
    f_inst = (Fs/(2*pi)) * dphi;
    % match original length with trailing NaN
    f_inst = [f_inst, NaN];
end
