function [stitchedData, triggers_std_analog_trimmed] = AV40_StitchContinuousData(continuous_raw, triggers_std_analog, desiredDuration, fs)
    % StitchContinuousData - Stitches epochs from continuous_raw into a continuous data matrix.
    % 
    % Inputs:
    %   continuous_raw        - Channels x Epochs x Time (Continuous data from AV40_Import.mat)
    %   triggers_std_analog   - Trigger times (in samples) corresponding to 30 kHz
    %   chunkStartTimes       - Start times of each epoch (in seconds)
    %   desiredDuration       - Duration to stitch in seconds
    %   fs                    - the fs of both the chunked and stitched data
    %
    % Outputs:
    %   stitchedData               - Channels x Time (stitched continuous data)
    %   triggers_std_analog_trimmed - Trimmed triggers to match the stitched data duration
    %
    % Example:
    %   [stitchedData, triggers_std_analog_trimmed] = StitchContinuousData(continuous_raw, triggers_std_analog, chunkStartTimes, 300, 1000);

    % Size of the continuous_raw matrix
    [numChannels, numChunks, chunkLength] = size(continuous_raw);

    % Compute total available duration in seconds
    totalDuration = numChunks; % Assuming each chunk is 1 second long
    
    % Check if requested duration exceeds available duration
    if desiredDuration > totalDuration
        error('Requested duration exceeds available data. Maximum available duration is %d seconds.', totalDuration);
    end

    % Determine how many chunks to stitch together
    numChunksToStitch = floor(desiredDuration);

    % Initialize stitchedData matrix
    stitchedData = [];

    for chunkIdx = 1:numChunksToStitch
        stitchedData = [stitchedData, squeeze(continuous_raw(:, chunkIdx, :))];
    end

    % Calculate stitched data duration in samples
    stitchedDuration_samples = size(stitchedData, 2);
    stitchedDuration_sec = stitchedDuration_samples / fs;

    % Trim triggers_std_analog to match stitched data duration
    triggers_std_analog_sec = triggers_std_analog / 30000; % Convert to seconds
    triggers_std_analog_trimmed = triggers_std_analog_sec(triggers_std_analog_sec <= stitchedDuration_sec);

    % Convert trimmed triggers back to samples
    triggers_std_analog_trimmed = round(triggers_std_analog_trimmed * 30000);

    % Display summary
    fprintf('Stitched Data Duration: %.2f seconds (%d samples)\n', stitchedDuration_sec, stitchedDuration_samples);
    fprintf('Number of triggers retained: %d\n', length(triggers_std_analog_trimmed));
end
