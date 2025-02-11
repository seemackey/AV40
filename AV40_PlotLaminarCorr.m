function [] = AV40_PlotLaminarCorr(results,selectedChannels)
% inputs
% the results struct from AV40_PerfAndNeuroOverTime
% and the selected channels for the laminar plot
% not outputs, just a figure


% Select relevant channels (2-22)
selectedChannels = 2:22;
alphaAmp = results.alphaAmp;
deltaITC = results.deltaITC;
dprime = results.dprime;
% Extract the data for selected channels
alphaAmpSel = alphaAmp(selectedChannels, :);
deltaITCSel = deltaITC(selectedChannels, :);
dprimeSel = dprime(:)'; % Ensure dprime is row vector



%%
% Select relevant channels (2-22) and reverse order for plotting
selectedChannels = 2:22;
selectedChannelsRev = flip(selectedChannels); % Reverse order

% Initialize arrays to store correlation values
rho_alpha = NaN(length(selectedChannels), 1);
p_alpha = NaN(length(selectedChannels), 1);
rho_delta = NaN(length(selectedChannels), 1);
p_delta = NaN(length(selectedChannels), 1);

% Compute correlation for each channel separately
for idx = 1:length(selectedChannels)
    ch = selectedChannels(idx);

    % Extract data for current channel
    alphaAmpVec = alphaAmp(ch, :)';
    deltaITCVec = deltaITC(ch, :)';
    dprimeVec = dprime(:); % Ensure dprime is column vector

    % **Alpha Amplitude Correlation**
    validIdx = ~isnan(alphaAmpVec) & ~isnan(dprimeVec);
    if sum(validIdx) > 2 % Ensure enough valid data points
        [rho_alpha(idx), p_alpha(idx)] = corr(alphaAmpVec(validIdx), dprimeVec(validIdx), 'Type', 'Spearman');
    end

    % **Delta ITC Correlation**
    validIdx = ~isnan(deltaITCVec) & ~isnan(dprimeVec);
    if sum(validIdx) > 2
        [rho_delta(idx), p_delta(idx)] = corr(deltaITCVec(validIdx), dprimeVec(validIdx), 'Type', 'Spearman');
    end
end

% Reverse order for plotting
% rho_alpha = flip(rho_alpha);
% rho_delta = flip(rho_delta);
% p_alpha = flip(p_alpha);
% p_delta = flip(p_delta);

% **? Plot Results**
figure;

plot(rho_alpha,selectedChannels, 'b'); hold on;
xlabel('Spearman Correlation');
ylabel('Channel');
title('Correlation of Accuracy with Oscillatory Activity');
xlim([-1 1]);
grid on;
set(gca, 'YDir', 'reverse');
% yticks(selectedChannelsRev);
% yticklabels(arrayfun(@num2str, selectedChannelsRev, 'UniformOutput', false));

hold on
plot(rho_delta, selectedChannels, 'r'); hold on;
legend('Alpha','Delta ITC')
% yticks(selectedChannelsRev);
% yticklabels(arrayfun(@num2str, selectedChannelsRev, 'UniformOutput', false));


end
