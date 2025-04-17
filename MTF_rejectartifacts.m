function [cleaned_data, outlier_idx_unique] = MTF_rejectartifacts(data, method, threshold)
    % Artifact rejection function
    % data: Input data (channels x trials x time)
    % method: Method to identify outliers ('median' or 'mean')
    % threshold: Threshold for identifying outliers

    if nargin < 2
        method = 'median';  % Default method
    end
    
    if nargin < 3
        threshold = 3;  % Default threshold (z-score)
    end

    num_trials = size(data, 2);

    max_vals = zeros(1, num_trials);

    % Calculate the maximum mean absolute value for each trial
    for trial = 1:num_trials
        max_vals(trial) = max(mean(abs(squeeze(data(:, trial, :))), 1));
    end

    % Identify outliers based on the chosen method
    switch method
        case 'median'
            outlier_idx = abs(max_vals - median(max_vals)) > threshold * std(max_vals);
        case 'mean'
            outlier_idx = abs(max_vals - mean(max_vals)) > threshold * std(max_vals);
        otherwise
            error('Unsupported method. Use "median" or "mean".');
    end

    outlier_idx_unique = find(outlier_idx);

    % Remove outliers
    cleaned_data = data;
    cleaned_data(:, outlier_idx_unique, :) = [];

end