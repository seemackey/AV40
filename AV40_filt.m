function [cnte, cntm] = AV40_filt(craw, newadrate, filtere, filteru, fs)
% MODULE_FILTCONT - Filter raw data into LFP and MUA bands and downsample.
%
% Inputs:
%   craw      - Raw data (1 channel × samples)
%   newadrate - New sampling rate
%   filtere   - LFP filter range [low, high] (Hz)
%   filteru   - MUA filter range [low, high] (Hz)
%   fs        - Original sampling rate (Hz)
%
% Outputs:
%   cnte      - Filtered LFP signal (1 × samples)
%   cntm      - Filtered and rectified MUA signal (1 × samples)

    % Ensure craw is a row vector
    if iscolumn(craw)
        craw = craw';
    end

    % LFP Filtering
    if filtere(1) == 0
        [b, a] = butter(2, filtere(2) / (fs / 2), 'low');
    else
        [b, a] = butter(2, filtere / (fs / 2));
    end
    cnte = filtfilt(b, a, craw);

    % MUA Filtering
    if filteru(2) == 0
        [b, a] = butter(2, filteru(1) / (fs / 2), 'high');
    else
        [b, a] = butter(2, filteru / (fs / 2));
    end
    cntu = filtfilt(b, a, craw);

    % Rectify MUA
    cntm = abs(hilbert(cntu));

    % Downsample to newadrate
    if fs ~= newadrate
        cnte = resample(cnte, newadrate, fs);
        cntm = resample(cntm, newadrate, fs);
    end
end
