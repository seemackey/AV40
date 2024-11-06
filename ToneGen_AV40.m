% Settings
frequencies = [500,707,1000,1414,2000,2828,4000,5656,8000,11300,16000];
amplitudes = [0.005,0.01,0.015,0.02,0.05,0.1];
dur = 0.1; % Duration in seconds
riseDur = 0.005; % Rise/fall duration in seconds
phase = 0; % Initial phase
Fs = 48000; % Sampling frequency

% Loop through each frequency and amplitude
for freq = frequencies
    for amp = amplitudes
        % Generate tone
        [tone, tVec, env] = genTone(amp, freq, dur, riseDur, phase, Fs);

        % Normalize tone to final amplitude
        tone = tone / max(abs(tone)) * amp;

        % Generate filename
        filename = sprintf('%d_amp_%.3f.wav', freq, amp);

        % Save as .wav file
        audiowrite(filename, tone, Fs);
        %figure;plot(tVec,tone)

        % Display confirmation
        fprintf('Generated file: %s\n', filename);
    end
end

% Function to generate tone with rise and fall envelope
function [tone, tVec, env] = genTone(amp, freq, dur, riseDur, phase, Fs)
    % Calculate number of points required
    dt = 1/Fs;
    nPts = round(dur * Fs);
    
    % Generate time vector
    tVec = (0:nPts-1) * dt;

    % Generate tone
    tone = amp * sin(2 * pi * freq * tVec + phase);

    % Generate rise/fall envelope
    nPtsRise = round(riseDur * Fs);
    if nPtsRise > nPts / 2
        nPtsRise = floor(nPts / 2);
    end
    tVecRise = linspace(-pi/2, 0, nPtsRise);

    % Create rise and fall sections
    rise = cos(tVecRise);
    fall = rise(end:-1:1);

    % Construct envelope
    env = ones(1, nPts);
    env(1:nPtsRise) = rise;
    env(end-(nPtsRise-1):end) = fall;

    % Apply envelope to tone
    tone = env .* tone;
end
