function out = corr_heatmap_over_time_simple(fileA, fileB, doPlot)
% Minimal across-recording OverTime correlation heatmaps.
% Uses fields: results.alphaAmp, results.deltaITC, results.gpITC
% Correlates channel×channel (Spearman) across common windows.
%
% Usage:
%   out = corr_heatmap_over_time_simple('recA_OverTime.mat','recB_OverTime.mat', true);

if nargin < 3, doPlot = true; end

S1 = load(fileA); R1 = S1.results;
S2 = load(fileB); R2 = S2.results;

metrics = { ...
    'alphaAmp',  R1.alphaAmp,  R2.alphaAmp; ...
    'deltaITC',  R1.deltaITC,  R2.deltaITC; ...
    'gpITC',     R1.gpITC,     R2.gpITC ...
};

out = struct();
out.meta = struct('fileA',fileA, 'fileB',fileB);

if doPlot
    f = figure('Name','Across-recording channel correlations (Spearman)','Position',[80 80 1400 950]);
    tl = tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
    clim = [-1 1];
end

for m = 1:size(metrics,1)
    name = metrics{m,1};
    A    = metrics{m,2};   % [nChanA x nWinA]
    B    = metrics{m,3};   % [nChanB x nWinB]

    % Trim to common window count (index-wise)
    nCommon = min(size(A,2), size(B,2));
    A = A(:,1:nCommon);
    B = B(:,1:nCommon);

        % 1) Drop all-NaN channels (rows) in each recording
    A = A(any(isfinite(A),2), :);
    B = B(any(isfinite(B),2), :);

    % 2) For cross-recording correlation, keep only windows with no NaN in either A or B
    validAB = all(isfinite(A),1) & all(isfinite(B),1);
    Aab = A(:, validAB);
    Bab = B(:, validAB);

    % For within-recording correlations, keep windows with no NaN in that recording
    validA = all(isfinite(A),1);
    validB = all(isfinite(B),1);
    Aaa = A(:, validA);
    Bbb = B(:, validB);

    A = Aaa;
    B = Bbb;

    % Spearman correlation across windows:
    % corr works column-wise -> put windows as rows (transpose)
    R_AB = corr(A', B', 'Type','Spearman', 'Rows','pairwise'); % [nChanA x nChanB]
    R_AA = corr(A',      'Type','Spearman', 'Rows','pairwise'); % [nChanA x nChanA]
    R_BB = corr(B',      'Type','Spearman', 'Rows','pairwise'); % [nChanB x nChanB]

    out.(name).R_AB = R_AB;
    out.(name).R_AA = R_AA;
    out.(name).R_BB = R_BB;
    out.(name).nCommonWindows = nCommon;

    if doPlot
        % A x B
        nexttile(tl); imagesc(R_AB, clim); axis image; colorbar;
        title(sprintf('%s ', name), 'Interpreter','none');
        xlabel('A1 channels'); ylabel('MGB channels');

        % A x A
        nexttile(tl); imagesc(R_AA, clim); axis image; colorbar;
        title(sprintf('%s  A×A', name), 'Interpreter','none');
        xlabel('A channels'); ylabel('A channels');

        % B x B
        nexttile(tl); imagesc(R_BB, clim); axis image; colorbar;
        title(sprintf('%s  B×B', name), 'Interpreter','none');
        xlabel('B channels'); ylabel('B channels');
    end
end

if doPlot
    sgtitle('Spearman channel×channel correlations across common windows');
end
end
