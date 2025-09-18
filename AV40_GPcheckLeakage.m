%% GP Pre-stim Leakage Check on CSD (example trials)
% chase m 2025 — quick diagnostic

clear; close all; clc;

%% ---------- Config ----------
pathName = 'H:\Peter\pt064065\AttAudCtx\';
fileName = 'pt064065023_analog_continuous';   % base file that also contains EyelinkData
eventType = 'ADDT';

fs        = 1000;             % Hz (continuous data & GP)
GP_band   = [1 100];           % Hz (wideband GP)
pre_ms    = 400;              % plot window pre-stim (ms)
post_ms   = 400;              % plot window post-stim (ms)
eval_ms   = -300;             % where to read the pre-stim phase (ms)
evoked_cut_ms = [0 300];      % zero-out window to remove evoked (ms)
selchan_mono  = 13;            % mono index whose CSD center you want
nTrialsToShow = 3;

%% ---------- Load ----------
disp('Loading EyelinkData & continuous...');
load(fullfile(pathName, fileName), 'EyelinkData', 'continuous_raw', 'triggers_std_analog');

% event times (s)
ADDT_STANDARD = EyelinkData.ADDT_STANDARD / fs;
ADDT_DEVIANT  = EyelinkData.ADDT_DEVIANT  / fs;
VDDT_STANDARD = EyelinkData.VDDT_STANDARD / fs;
VDDT_DEVIANT  = EyelinkData.VDDT_DEVIANT  / fs;
VST_ONSET     = EyelinkData.VST_ONSET     / fs;

% align to analog triggers
rippleFs = 30000;
firstEyelinkStandard = min(ADDT_STANDARD);
firstNeuralStandard  = min(triggers_std_analog)/rippleFs;
timeOffset = firstNeuralStandard - firstEyelinkStandard;

% choose event series
switch eventType
    case 'ADDT'
        eventTimes = (triggers_std_analog / rippleFs); % use neural triggers for standards
    case 'VDDT'
        eventTimes = VDDT_STANDARD + timeOffset;
    case 'VST'
        eventTimes = VST_ONSET + timeOffset;
    otherwise
        error('Invalid eventType.');
end
eventTimes = eventTimes(:);

%% ---------- Build continuous CSD (one target channel) ----------
% continuous_raw: [nChan x nChunks x 1000]
[nMono, nChunks, nSamp] = size(continuous_raw);
T = nChunks * nSamp;

% CSD center at selchan_mono -> requires mono (sel-1:sel+1)
if selchan_mono < 2 || selchan_mono > (nMono-1)
    error('selchan_mono must be in [2, nMono-1] for CSD center.');
end
chTri = selchan_mono-1 : selchan_mono+1;

% reshape to [3 x T]
mono3 = reshape(permute(continuous_raw(chTri,:,:), [1 3 2]), 3, T);

% filter each mono channel once (0.5–300) like your pipeline
mono3_filt = zeros(size(mono3));
for k = 1:3
    [mono3_filt(k,:), ~] = AV40_filt(mono3(k,:), fs, [0.5 300], [300 400], fs);
end

% CSD (single row) centered at selchan: -diff^2 over depth
csd_ts = -diff(mono3_filt, 2, 1);  % -> [1 x T]
csd_ts = csd_ts(:).';              % row vector

%% ---------- GP (normal, on full signal) ----------
phi_full = gp_phase_acausal(csd_ts, fs, 1, GP_band);   % 1 s reflect pad inside

%% ---------- Pick trials with margin ----------
preS  = round(pre_ms  * fs / 1000);
postS = round(post_ms * fs / 1000);
evalS = round(eval_ms * fs / 1000);
cut0  = round(evoked_cut_ms(1)*fs/1000);
cut1  = round(evoked_cut_ms(2)*fs/1000);

% Keep trials fully inside the file with the required margins
evtIdx = round(eventTimes*fs);
good = find(evtIdx-preS > 0 & evtIdx+postS <= T & evtIdx+cut1 <= T);
if isempty(good)
    error('No events with adequate pre/post margins.');
end
good = good(1:min(nTrialsToShow, numel(good)));

%% ---------- Plot a few trials ----------
t_rel = (-preS:postS)/fs*1000;           % ms relative to stim
t_rel_pre = (-preS:0)/fs*1000;

figure('Name','CSD vs GP pre-stim phase (leakage check)','Position',[80 80 1200 800]);
tiledlayout(numel(good), 2, 'TileSpacing','compact','Padding','compact');

for ii = 1:numel(good)
    s0 = evtIdx(good(ii));             % stim sample
    seg = csd_ts(s0-preS : s0+postS);  % CSD segment

    % GP phase from full signal (acausal, potential leakage)
    segPhi_pre = phi_full(s0-preS : s0);    % only pre-stim part for display
    phi_eval_full = phi_full(s0 + evalS);   % at eval_ms (e.g., -300 ms)

    % Build "evoked-removed" signal for this trial and recompute GP
    x_mod = csd_ts;
    x_mod(s0+cut0 : s0+cut1) = 0;      % zero post-stim window [0..+cut1]
    phi_mod = gp_phase_acausal(x_mod, fs, 1, GP_band);
    segPhi_pre_mod = phi_mod(s0-preS : s0);
    phi_eval_mod = phi_mod(s0 + evalS);

    % circular phase difference (degrees) at eval_ms
    dphi = angle(exp(1i*(phi_eval_full - phi_eval_mod)));
    dphi_deg = rad2deg(dphi);

    % --- Plot CSD ---
    nexttile; 
    plot(t_rel, seg, 'k'); hold on;
    xline(0,'r-'); yline(0,'k:');
    title(sprintf('Trial %d  (CSD, ch %d center)', good(ii), selchan_mono));
    xlabel('Time rel. stim (ms)'); ylabel('CSD (a.u.)'); grid on;

    % --- Plot GP phase (pre) full vs evoked-removed ---
    nexttile;
    plot(t_rel_pre, wrapToPi(segPhi_pre), 'b', 'LineWidth', 1.2); hold on;
    plot(t_rel_pre, wrapToPi(segPhi_pre_mod), 'g--', 'LineWidth', 1.2);
    xline(eval_ms,'m:','-300 ms'); xline(0,'r-');
    ylim([-pi pi]); yticklabels({'-π','-π/2','0','π/2','π'});
    title(sprintf('\\phi_{full} vs \\phi_{no-evoked}  @%d ms: \\Delta=%.1f°', eval_ms, dphi_deg));
    xlabel('Time rel. stim (ms)'); ylabel('Phase (rad)'); grid on;
    legend({'full','no-evoked'},'Location','southoutside','Orientation','horizontal');
end

sgtitle(sprintf('Pre-stim GP phase leakage check (GP %d-%d Hz) — eval at %d ms', GP_band(1), GP_band(2), eval_ms));

%% ---------- Helper: acausal GP with reflect padding ----------
function phi = gp_phase_acausal(x, Fs, pad_sec, GP_band)
    x = double(x(:));  T = numel(x);
    pad = max(1, round(pad_sec*Fs));
    pre  = flipud(x(1:min(pad, T)));
    post = flipud(x(max(1, end-min(pad,T)+1):end));
    xpad = [pre; x; post];

    % band-limit (zero-phase)
    [b,a] = butter(4, GP_band/(Fs/2));
    xpadf = filtfilt(b,a,xpad);

    % generalized phase (Müller/Davis)
    xgp = generalized_phase_vector(xpadf, Fs, 0);
    phip = angle(xgp(:));

    % slice center back to original length
    phi = phip(numel(pre)+(1:T));
end
