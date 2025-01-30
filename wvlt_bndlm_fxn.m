function [amplitude,phase] = wvlt_bndlm_fxn(data,Fs,frequencyRange,targetBand)

    %%  Wavelet Transform, average power in a band, phase

    % Initialization and Load Data
    %clear; close all;
    %load('path_to_your_time_series_data.mat'); % Load your time-domain signal data here


    % Compute the Continuous Wavelet Transform
    [wt, f] = cwt(data, Fs, 'FrequencyLimits', frequencyRange);

    % Isolate the target frequency band (8-14 Hz)
    targetIdx = f >= targetBand(1) & f <= targetBand(2);
    wtTargetBand = wt(targetIdx, :);

    % sum along freq dimension
    wtTargetBandSum = sum(wtTargetBand,1);

    % Compute amplitude and phase for the target band
    amplitude = abs(wtTargetBandSum);
    phase = real(wtTargetBandSum); % angle could be used here, using real instead


end

