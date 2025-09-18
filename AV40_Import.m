% import ephys from the ripple system and eyelink data from txt files 
% chase m 2024
% epochs to digital triggers and eyelink data, feb '25
% sorts trigger types from event files, march '25
% fixed digital triggering from TDT programs, may '25

% Clear workspace and initialize
clear;
close all;
clc;

% Directories and file names

inputDir = 'H:\Peter\pt066067\'; % Replace with your input directory
figuresDir = 'H:\Peter\pt066067\AttAudCtx'; % Replace with your output directory
fileName = 'pt066067024.nev'; % Replace with your file name
 % Replace with your file name
% inputDir = '/Volumes/Samsung03/data/AM/'; % Replace with your input directory
% figuresDir = '/Volumes/Samsung03/data/AM/'; % Replace with your output directory
% fileName = 'ke036037037.nev'; % Replace with your file name
%"H:\Peter\pt060061031.nev"


%/Volumes/Samsung03/data/AM/ke036037037.nev


% Example configuration
config = struct();
config.epochdata = 1; % boolean, want to epoch the data?
config.epoch_tframe = [-50, 150]; % Epoch window in ms
config.ripplefs = 30000; % assumed ripple fs/adrate
config.eyelinkfs = 1000; %assumed eyelink FS
config.newadrate = 1000;          % Resampling rate
config.filters.lfp = [0.5, 300];  % LFP filter range (Hz)
config.filters.mua = [300, 5000]; % MUA filter range (Hz)
config.padding = 1000; % ms of padding for epoched data
config.derivative = 2; % CSD is second (2) deriv, bipLFP is 1st deriv (1)
config.trigger_channel = 29;  %  analog trigger channel for aud is hardcoded at 29 for now, sorry dear reader, 9mgb, 46ctx
config.channels = [33:56];             % ephys data channels 1:24 or 33:56
config.channel_remap = true; % Enable ripple channel remapping 
config.trigger_method = 'analog'; % 'digital', 'analog', 'segment' for estim, 'VDDT', 'VST' *always use analog for aud epoching*
config.segEntID = 1;
config.trigger_threshold = 50;   % Threshold for analog trigger detection, 50 for aud, 5000 for estim artifact
config.event_entity_id = 1;       % Default Event Entity ID
config.artifact_threshold = 3;    % Z-score threshold for artifact rejection
config.checksync = 0; % check sync between ripple and eyelink, boolean
config.get_deviant = 0; %boolean
config.event_sorting_method = [];%'oldtono';%'ev2_column';%'ev2_column'; % a string that picks 'oldtono' or an ev2 column; requires digital trigger method
config.durationForTuningCurve = 50;
config.selectedVariable = [];%'Modulation_Freq';%'Modulation_Freq'; % variable in event file to epoch to;requires digital trigger method
config.store_cont_data = 1; %takes a while, stores 1s chunks@newadrate

% new artifact section to interpolate over artifacts (e.g. e-stim onsets)
config.artifact.enabled    = false;
config.artifact.windows_ms = [-3 3];   % one or more [start end] ms windows
config.artifact.pad_ms     = 1;                % optional padding
config.artifact.method     = 'linear';         % or 'pchip'
config.artifact.signals    = {'mua'};    % subset (default = all present)


%% MAIN FUNCTION CALL
try
    % Call the main data import function
    if ~exist(figuresDir, 'dir')
     mkdir(figuresDir);
    end
    [epoched_data] = data_import_v2(inputDir,figuresDir, fileName, config);
    
    
    
    %% make figs
    if config.epochdata == 1
    plot_baseline_corrected_data(epoched_data.standard, config, 'standard', figuresDir, fileName);
    end
    % Plot deviant
    if config.get_deviant == 1
        plot_baseline_corrected_data(epoched_data.deviant, config, 'deviant', figuresDir, fileName);
    end
    
    % plot sorted trig types
    if ~isempty(config.event_sorting_method)
        if contains(config.event_sorting_method,'oldtono')
           plot_baseline_corrected_data_mult_trigType(epoched_data, config, figuresDir, fileName)

        end
    end
    
catch ME
    % Handle errors 
    fprintf('An error occurred: %s\n', ME.message);
end

%% FUNCTIONS BELOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [epoched_data] = data_import_v2(directory1, figuresDir, fileName, config)
    % DATA_IMPORT_V2 - Trigger-based Import 
    
    %% Open File
    filePath = fullfile(directory1, fileName);
    [ns_RESULT, hFile] = ns_OpenFile(filePath);
    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Failed to open file: %s', ns_RESULT);
    end
    
    %% classify channels and REMAP according to labels
    % Separate entities into raw (data) and analog (stimuli and eye mvmt) categories
    [rawChannels, analogChannels, digitalChannels] = classify_channels(hFile); % raw and ana chans
    
    config.channels = rawChannels(config.channels); % selected raw chans
    %% Get Trigger Times
    
    if contains(config.trigger_method,'analog') % recorded audio
 
        triggerChannel = analogChannels(config.trigger_channel);
        [triggers_std, triggers_deviant] = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold);
        triggers_std_analog = triggers_std;
        % Ensure column vectors
        triggers_std = triggers_std(:);
        triggers_deviant = triggers_deviant(:);

        % Get Eyelink data
        EyelinkData = AV40_importEyelink(directory1, fileName);


     elseif contains(config.trigger_method,'digital') % recorded TTLs
        %if isempty(config.event_sorting_method)
        % --- Digital TTL Trigger Extraction ---
        triggers_std = [];
        all_TTLs = [];
        all_timestamps = [];

        numEvents = hFile.Entity(config.event_entity_id).Count;

        % Extract all timestamps and TTL values
        for eventIdx = 1:numEvents
            [~, timestamp, TTL_data, ~] = ns_GetEventData(hFile, config.event_entity_id, eventIdx);
            all_timestamps = [all_timestamps; timestamp];
            all_TTLs = [all_TTLs; TTL_data];

        end

        % Define TTL "high" values (adjust threshold as needed)
        isHigh = all_TTLs >= 1000;  % Catch only pulses (e.g., 32767)

        % Extract timestamps at rising edges only
        triggers_std = all_timestamps(isHigh);
        isi = diff(triggers_std);
        isi_real = isi > 0.009; 
        badTrigIdx =  find(isi_real == 0);
        triggers_std(badTrigIdx) = [];

        % Convert to Ripple sample rate (30 kHz)
        triggers_std = round(triggers_std(:) * config.ripplefs);


        triggers_std_analog = [];
        triggers_deviant = [];

        % handle new and old event file types here
        baseName = erase(fileName, '.nev'); % clean filename
        datDir = dir(fullfile(directory1, '*@v*')); % DAT files (old tono method)
        datDir = datDir(~startsWith({datDir.name}, '._')); % remove this crap if on a mac
        ev2Dir = dir(fullfile(directory1, '*.ev2')); % EV2 files
        ev2Dir = ev2Dir(~startsWith({ev2Dir.name}, '._')); % remove this crap if on a mac
        
        
            if ~isempty(config.event_sorting_method)
                if contains(config.event_sorting_method, 'oldtono')
                    % -- Method: oldtono (e.g., DAT file with @v) --
                    datFileMatch = contains({datDir.name}, baseName);




                    if ~any(datFileMatch)
                        error('No matching DAT file (@v) found for file %s', baseName);
                    end
                    datFile = load(fullfile(datDir(datFileMatch).folder, datDir(datFileMatch).name));

                    % Assume first column contains event IDs
                    trigType = datFile(:, 1); % adjust fieldname if needed

                    if length(trigType) ~= length(triggers_std)
                        warning('Mismatch: trigIDs (%d) ? triggers (%d)', length(trigType), length(triggers_std));
                        %triggers_std = triggers_std(1:length(trigType));
                    end

                    triggers_deviant = NaN; % not applicable
                    %trigger_metadata.labels = toneIDs;

                elseif contains(config.event_sorting_method, 'ev2_column') && ~isempty(config.event_sorting_method)
                    % Find the ev2 in our current directory
                    ev2Match = contains({ev2Dir.name}, baseName);
                    if ~any(ev2Match)
                        error('No matching EV2 file found for file %s', baseName);
                    end
                    ev2FilePath = fullfile(ev2Dir(ev2Match).folder, ev2Dir(ev2Match).name);

                    % get out the whole event file, or just a selected column
                    [~,selectedColumn] = ev2LoadAndParse(ev2FilePath,config.selectedVariable);

                    % Extract condition labels for triggers
                    trigType = selectedColumn;

                    % old way
    %                 ev = dlmread(ev2FilePath, ' ');
    %                 col6=ev(:,6);
    %                  trigType = col6;

                    % Ensure number of trials matches number of triggers
                    if length(conditionVals) ~= length(triggers_std)
                        warning('Mismatch: EV2 trials (%d) ? triggers (%d)', length(trigType), length(triggers_std));
                        triggers_std = triggers_std(1:length(trigType));
                    end

                    triggers_deviant = NaN;
                end
            end
        
        
    elseif contains(config.trigger_method, 'segment')
        fprintf('Using segment entity %d as triggers\n', config.segEntID);
    
        
        numSegments = hFile.Entity(config.segEntID).Count;
    
        triggers_std = zeros(numSegments, 1);
    
        for i = 1:numSegments
            [~, ts, ~, ~, ~] = ns_GetSegmentData(hFile, config.segEntID, i);
            triggers_std(i) = ts;
        end
    
        triggers_std = round(triggers_std(:) * config.ripplefs);  % convert to samples
        triggers_deviant = [];
        triggers_std_analog = [];
    elseif contains(config.trigger_method,'stimArtifact') % find stim artifact
 
        triggerChannel = config.trigger_channel;
        [triggers_std] = get_segment_triggers(hFile, triggerChannel, config.trigger_threshold);
        triggers_std_analog = triggers_std;
        % Ensure column vectors
        triggers_std = triggers_std(:);
        triggers_deviant = [];
        
   
    elseif contains(config.trigger_method,'ADDT')
    
    
        %% Eyelink-Based Triggering (unreliable for Aud signals)
        % only use as redundant backup for analog triggering
        EyelinkData = AV40_importEyelink(directory1, fileName); % Eyelink txt
        triggerChannel = analogChannels(config.trigger_channel);
        [triggers_std_analog, ~] = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold); % Recorded audio

        %Convert Eyelink & Neural Data to Samples 
        firstEyelinkStandard_sample = min(EyelinkData.ADDT_STANDARD); % Already in 1 kHz samples
        firstNeuralStandard_sample = round(min(triggers_std_analog) * (config.eyelinkfs / config.ripplefs)); % Convert to 1 kHz sample space

        %Compute Time Offset in **Samples**
        timeOffset_samples = firstNeuralStandard_sample - firstEyelinkStandard_sample; % Offset in 1 kHz sample space
        fprintf('Time Offset (Samples @1kHz): %d samples\n', timeOffset_samples);

        %Apply Offset to Eyelink Data in Sample Space
        triggers_std_samples = EyelinkData.ADDT_STANDARD + timeOffset_samples;
        triggers_deviant_samples = EyelinkData.ADDT_DEVIANT + timeOffset_samples;

        %Convert to Neural Sample Space (30 kHz)
        triggers_std = round(triggers_std_samples * (config.ripplefs / config.eyelinkfs));
        triggers_deviant = round(triggers_deviant_samples * (config.ripplefs / config.eyelinkfs));

         % Handle empty deviant triggers
        if isempty(triggers_deviant)
            triggers_deviant = [];
        end
        
    elseif contains(config.trigger_method,'VDDT')
    
        EyelinkData = AV40_importEyelink(directory1, fileName); % Eyelink txt
        triggerChannel = analogChannels(config.trigger_channel);
        [triggers_std_analog, ~] = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold); % Recorded audio

        %Convert Eyelink & Neural Data to Samples First
        firstEyelinkStandard_sample = min(EyelinkData.ADDT_STANDARD); % Already in 1 kHz samples
        firstNeuralStandard_sample = round(min(triggers_std_analog) * (config.eyelinkfs / config.ripplefs)); % Convert to 1 kHz sample space

        %Compute Time Offset in **Samples**
        timeOffset_samples = firstNeuralStandard_sample - firstEyelinkStandard_sample; % Offset in 1 kHz sample space
        fprintf('Time Offset (Samples @1kHz): %d samples\n', timeOffset_samples);

        %Apply Offset to Eyelink Data in Sample Space
        triggers_std_samples = EyelinkData.VDDT_STANDARD + timeOffset_samples;
        triggers_deviant_samples = EyelinkData.VDDT_DEVIANT + timeOffset_samples;

        %Convert to Neural Sample Space (30 kHz)
        triggers_std = round(triggers_std_samples * (config.ripplefs / config.eyelinkfs));
        triggers_deviant = round(triggers_deviant_samples * (config.ripplefs / config.eyelinkfs));


        % Handle empty deviant triggers
        if isempty(triggers_deviant)
            triggers_deviant = [];
        end

    elseif contains(config.trigger_method,'VST')

                EyelinkData = AV40_importEyelink(directory1, fileName); % Eyelink txt
        triggerChannel = analogChannels(config.trigger_channel);
        [triggers_std_analog, ~] = get_analog_triggers(hFile, triggerChannel, config.trigger_threshold); % Recorded audio

        %Convert Eyelink & Neural Data to Samples 
        firstEyelinkStandard_sample = min(EyelinkData.ADDT_STANDARD); % Already in 1 kHz samples
        firstNeuralStandard_sample = round(min(triggers_std_analog) * (config.eyelinkfs / config.ripplefs)); % Convert to 1 kHz sample space

        %Compute Time Offset in **Samples**
        timeOffset_samples = firstNeuralStandard_sample - firstEyelinkStandard_sample; % Offset in 1 kHz sample space
        fprintf('Time Offset (Samples @1kHz): %d samples\n', timeOffset_samples);

        %Apply Offset to Eyelink Data in Sample Space
        triggers_std_samples = EyelinkData.ADDT_STANDARD + timeOffset_samples;
        %triggers_deviant_samples = EyelinkData.ADDT_DEVIANT + timeOffset_samples;

        %Convert to Neural Sample Space (30 kHz)
        triggers_std = round(triggers_std_samples * (config.ripplefs / config.eyelinkfs));
        %triggers_deviant = round(triggers_deviant_samples * (config.ripplefs / config.eyelinkfs));
                % Handle empty deviant triggers
        % std and dev are simultaneous in VS condition
            triggers_deviant = [];
    else
        disp('good grief. check how you set trigger method in config')
    end

        
    

    %% Sampling Rates and Epoch Setup
    fs = config.ripplefs; % Ripple sampling rate (30 kHz for Ripple system)
    newadrate = config.newadrate; % Target downsampling rate (e.g., 1 kHz)

    % **Convert epoch window from milliseconds to samples (30 kHz)**
    x1 = round(config.epoch_tframe(1) * (fs / 1000)); % Convert to 30 kHz samples
    x2 = round(config.epoch_tframe(2) * (fs / 1000)); % Convert to 30 kHz samples

    % **Ensure all triggers are valid and in 30 kHz sample space**
    triggers_std = triggers_std(~isnan(triggers_std)); % Remove NaNs
    triggers_deviant = triggers_deviant(~isnan(triggers_deviant)); % Remove NaNs

    % **Compute Epoch Lengths**
    %epochLength_orig = abs(x1) + abs(x2) + 1; % Epoch length at **30 kHz**
    %epochLength_ds = round(epochLength_orig * (newadrate / fs)); % Downsampled epoch length (e.g., 1 kHz)
    epochLength_ds = round((x2 - x1) * (newadrate / fs)) + 1;

    %% **Pre-allocate Epoch Matrices**
    numChannels = length(config.channels);
    numStdTriggers = length(triggers_std);
    numDevTriggers = length(triggers_deviant);

    % Initialize  and add a space (+1) for the trigger
    lfp_std = NaN(numChannels, numStdTriggers, epochLength_ds);
    mua_std = NaN(numChannels, numStdTriggers, epochLength_ds);
    lfp_dev = NaN(numChannels, numDevTriggers, epochLength_ds);
    mua_dev = NaN(numChannels, numDevTriggers, epochLength_ds);

    
    %% Epoch Data for standards 
if config.epochdata == 1
        disp('Importing triggered data for standards...');
        for chIdx = 1:numChannels
            ch = config.channels(chIdx); % Select correct channel
            
            padMs = config.padding;
            padSamples = round((padMs / 1000) * fs);  % Padding in samples

            for tIdx = 1:numStdTriggers
                trigger = triggers_std(tIdx);

                % Expand window for padding
                paddedStart = max(round(trigger) + x1 - padSamples, 1);
                paddedEnd   = min(round(trigger) + x2 + padSamples, hFile.Entity(ch).Count);

                % Epoch window after padding (still used for trimming later)
                startIndex = round(trigger) + x1;
                endIndex   = round(trigger) + x2;

                % Safety check
                if paddedStart >= paddedEnd
                    warning('Trigger %d on channel %d is out of bounds. Skipping...', tIdx, ch);
                    continue;
                end

                % Request padded data from file
                paddedLength = paddedEnd - paddedStart + 1;
                [ns_RESULT, ~, epochDataPadded] = ns_GetAnalogData(hFile, ch, paddedStart, paddedLength);

                if strcmp(ns_RESULT, 'ns_OK') && numel(epochDataPadded) == paddedLength
                    % Filter the padded data
                    [lfp_padded, mua_padded] = AV40_filt(epochDataPadded, newadrate, config.filters.lfp, config.filters.mua, fs);

                    % Compute relative indices to slice out the core (unpadded) window
                    sliceStart = round((startIndex - paddedStart) * (newadrate / fs)) + 1;
                    sliceEnd   = round((endIndex   - paddedStart) * (newadrate / fs)) + 1;


                    % Ensure safe slicing
                    if sliceEnd > length(lfp_padded)
                        warning('Slice exceeds padded length. Skipping trigger %d on channel %d.', tIdx, ch);
                        continue;
                    end

                    % Extract central (unpadded) filtered window
                    lfp = lfp_padded(sliceStart:sliceEnd);
                    mua = mua_padded(sliceStart:sliceEnd);

                    % Store
                    lfp_std(chIdx, tIdx, :) = lfp;
                    mua_std(chIdx, tIdx, :) = mua;
                else
                    warning('Epoching failed for trigger %d on channel %d. Data mismatch.', tIdx, ch);
                    continue;
                end
            end

        end







    %% Remap Channels (if enabled)
    
        lfp_std = remap_channels(lfp_std);
        mua_std = remap_channels(mua_std);
    
    
    %% Compute CSD or Bipolar Signals
    
    deriv = config.derivative;
    if deriv == 1 % first derivative, AKA bipolar FP
        csd_std = -diff(lfp_std, 1, 1);
        lfp_std = lfp_std(1:end-1,:,:);
        mua_std = mua_std(1:end-1,:,:);
    else % second derivative, AKA the current source density
        csd_std = -diff(lfp_std, 2, 1);
        lfp_std = lfp_std(2:end-1,:,:); % match channels for convenience
        mua_std = mua_std(2:end-1,:,:);
    end

    %% Epoch Data for Deviants
    disp('Importing triggered data for deviants...');

        for chIdx = 1:numChannels
            ch = config.channels(chIdx); % Select correct channel
            
            padMs = config.padding;
            padSamples = round((padMs / 1000) * fs);  % Padding in samples

            for tIdx = 1:numDevTriggers
                trigger = triggers_deviant(tIdx);

                % Expand window for padding
                paddedStart = max(round(trigger) + x1 - padSamples, 1);
                paddedEnd   = min(round(trigger) + x2 + padSamples, hFile.Entity(ch).Count);

                % Epoch window after padding (still used for trimming later)
                startIndex = round(trigger) + x1;
                endIndex   = round(trigger) + x2;

                % Safety check
                if paddedStart >= paddedEnd
                    warning('Trigger %d on channel %d is out of bounds. Skipping...', tIdx, ch);
                    continue;
                end

                % Request padded data from file
                paddedLength = paddedEnd - paddedStart + 1;
                [ns_RESULT, ~, epochDataPadded] = ns_GetAnalogData(hFile, ch, paddedStart, paddedLength);

                if strcmp(ns_RESULT, 'ns_OK') && numel(epochDataPadded) == paddedLength
                    % Filter the padded data
                    [lfp_padded, mua_padded] = AV40_filt(epochDataPadded, newadrate, config.filters.lfp, config.filters.mua, fs);

                    % Compute relative indices to slice out the core (unpadded) window
                    sliceStart = round((startIndex - paddedStart) * (newadrate / fs)) + 1;
                    sliceEnd   = round((endIndex   - paddedStart) * (newadrate / fs)) + 1;


                    % Ensure safe slicing
                    if sliceEnd > length(lfp_padded)
                        warning('Slice exceeds padded length. Skipping trigger %d on channel %d.', tIdx, ch);
                        continue;
                    end

                    % Extract central (unpadded) filtered window
                    lfp = lfp_padded(sliceStart:sliceEnd);
                    mua = mua_padded(sliceStart:sliceEnd);

                    % Store
                    lfp_dev(chIdx, tIdx, :) = lfp;
                    mua_dev(chIdx, tIdx, :) = mua;
                else
                    warning('Epoching failed for trigger %d on channel %d. Data mismatch.', tIdx, ch);
                    continue;
                end
            end

        end
end

%% Sort unique trigger types (if applicable)
% ev = dlmread(epath, ' ');
% col6=ev(:,6);
% trigType = col6;

if ~isempty(config.event_sorting_method)
    if length(unique(trigType(~isnan(trigType))))>1 % unique trigs mean we sort the epoched data
    % Unique stimulus/trial condition types
    uniqueTypes = unique(trigType);
    
    % NaN is used by some programs to indicate broadband noise
    % so let's sort the NaNs too
    
    if sum(isnan(uniqueTypes))>0 % NaN detected
        nantypesidx = isnan(uniqueTypes); % get their indices
        firstNaN = min(find(nantypesidx==1));
        uniqueTypes = uniqueTypes(1:firstNaN);
    end
        
        
    nCond = length(uniqueTypes);
    
    
        
    % Initialize output struct
    sortedData = struct();
    disp('Sorting unique trigger types')

        for condIdx = 1:nCond
            thisType = uniqueTypes(condIdx);

            % Logical index of trials matching this condition
            if ~isnan(thisType)
                trialMask = trigType == thisType;
            else
                trialMask = isnan(trigType);
            end

            % Extract trials from each data array
            sortedData(condIdx).trigType = thisType;
            sortedData(condIdx).lfp = lfp_std(:, trialMask, :);
            sortedData(condIdx).csd = csd_std(:, trialMask, :);
            sortedData(condIdx).mua = mua_std(:, trialMask, :);

            % Optional: Store trial indices for debugging
            sortedData(condIdx).trialIndices = find(trialMask);
        end


        
    end
end
    
    
    %% Remap and Compute CSD for Deviants
    if config.epochdata == 1
    if config.channel_remap
        lfp_dev = remap_channels(lfp_dev);
        mua_dev = remap_channels(mua_dev);
    end
    
    deriv = config.derivative;
    if deriv == 1 % first derivative AKA bipolar lfp
        csd_dev = -diff(lfp_dev, 1, 1);
        lfp_dev = lfp_dev(1:end-1,:,:);
        mua_dev = mua_dev(1:end-1,:,:);
    else
        csd_dev = -diff(lfp_dev, 2, 1);
        lfp_dev = lfp_dev(2:end-1,:,:); % match channels for convenience
        mua_dev = mua_dev(2:end-1,:,:);
    end
    end
    
    %% downsample triggers before saving
    
    triggers_std_ds = round(triggers_std * (config.newadrate / fs));
    triggers_deviant_ds = round(triggers_deviant * (config.newadrate / fs));


    
    %% Import Continuous Data in 1-Second Chunks (Downsampled to 1 kHz)
    if config.store_cont_data == 1
    disp('Importing and downsampling continuous raw data into 1-second chunks...');

    chunkSize = fs; % 1-second chunk at original sampling rate (30 kHz)
    chunkSize_ds = config.newadrate; % Downsampled size (e.g. 1 kHz per second)
    totalSamples = hFile.Entity(config.channels(1)).Count; % Total samples in each channel

    numChunks = ceil(totalSamples / chunkSize);

    % Pre-allocate downsampled continuous data
    continuous_raw = NaN(numChannels, numChunks, chunkSize_ds);
    read_me_plz = 'chunk size for cont data is 1 second';
    for chIdx = 1:numChannels
        ch = config.channels(chIdx);
        for chunkIdx = 1:numChunks
            % Define chunk start and end indices
            startSample = (chunkIdx - 1) * chunkSize + 1;
            endSample = min(chunkIdx * chunkSize, totalSamples);
            chunkLength = endSample - startSample + 1;

            % Ensure valid chunk range
            if startSample <= totalSamples
                try
                    % Retrieve raw data chunk
                    [ns_RESULT, ~, chunkData] = ns_GetAnalogData(hFile, ch, startSample, chunkLength);
                    if strcmp(ns_RESULT, 'ns_OK') && ~isempty(chunkData)
                        % Downsample the chunk
                        chunkData_ds = resample(chunkData, chunkSize_ds, chunkLength);
                        continuous_raw(chIdx, chunkIdx, 1:length(chunkData_ds)) = chunkData_ds';
                    else
                        warning('Failed to retrieve or downsample chunk %d on channel %d.', chunkIdx, ch);
                    end
                catch ME
                    warning('Error processing chunk %d on channel %d: %s', chunkIdx, ch, ME.message);
                end
            else
                warning('Chunk %d on channel %d out of bounds. Skipping...', chunkIdx, ch);
            end
        end
    end
    
    
    % remap raw channels 
    continuous_raw = remap_channels(continuous_raw);
    
    % Save Continuous Raw Data
    
    disp('Continuous raw data imported and downsampled successfully.');
    else
        continuous_raw = [];
        read_me_plz = 'continuous data was not stored';
    end

    
    %% Save Data
    
    
    % Construct base filename without extensions
    fileBaseNameExt = sprintf('%s_%s', fileName, config.trigger_method);
    fileBaseName = regexprep(fileBaseNameExt, '\.nev|\.mat', '');
    
    %% Save Epoched data
    if config.epochdata == 1
        epoched_data.standard.lfp = lfp_std;
        epoched_data.deviant.lfp = lfp_dev;
        epoched_data.standard.csd = csd_std;
        epoched_data.deviant.csd = csd_dev;
        epoched_data.standard.mua = mua_std;
        epoched_data.deviant.mua = mua_dev;
        if exist('sortedData','var')
            epoched_data.sortedData = sortedData;
        end


        % Ensure EyelinkData exists before saving imported file
        importedFilePath = fullfile(figuresDir, [fileBaseName '_imported.mat']);




        if exist('EyelinkData', 'var')
            save(importedFilePath, 'epoched_data', 'triggers_std_ds', 'triggers_deviant_ds', 'EyelinkData','triggers_std_analog', 'config','-v7.3');
        else
            warning('EyelinkData not found. Saving without EyelinkData...');
            save(importedFilePath, 'epoched_data', 'triggers_std_ds', 'triggers_deviant_ds','triggers_std_analog', 'config','-v7.3');
        end
    
    else
        epoched_data = [];
    end
    
    %% Save Continuous unfiltered data
    % Ensure continuous file has the exact same base name, only replacing "_imported" with "_continuous"
    continuousFileBase = fullfile(figuresDir, [fileBaseName '_continuous.mat']);

    % Save continuous data only if store_cont_data flag is set
    
    if config.store_cont_data == 1 
        if exist('EyelinkData', 'var')
           
            save(continuousFileBase, 'continuous_raw','EyelinkData', 'triggers_std_ds', 'triggers_deviant_ds', 'triggers_std_analog', 'config', '-v7.3');
        else
            save(continuousFileBase, 'continuous_raw', 'triggers_std_ds', 'triggers_deviant_ds', 'triggers_std_analog', 'config', '-v7.3');
        end
    end


    disp('Data import complete!');
end




    




function plot_baseline_corrected_data(epoched_data, config, condition, figuresDir, fileName)
% PLOT_BASELINE_CORRECTED_DATA - Baseline-correct, optionally interpolate artifacts, and visualize epoched data
%
% Parameters:
% epoched_data - Struct with fields .lfp, .csd, .mua (dims: channels x trials x time)
% config       - Struct with fields:
%                  .epoch_tframe    [tStartMS tEndMS]
%                  .newadrate       sampling rate (Hz)
%                  .trigger_method  string used in filenames
%                Optional sub-struct for artifact interpolation:
%                  .artifact.enabled      (logical) default: false
%                  .artifact.windows_ms   (Nx2) [tStart tEnd] in ms, relative to epoch
%                  .artifact.pad_ms       (scalar) pad added to each side (default: 0)
%                  .artifact.method       'linear' (default) or 'pchip'
%                  .artifact.signals      cellstr subset of {'lfp','csd','mua'} (default: all present)
% condition    - 'standard' | 'deviant' (used for labels)
% figuresDir   - directory to save figures
% fileName     - base filename (no extension)

    disp(['Making figures for ', condition, ' data']);

    %----- Time axis from data size (robust to sampling rate / integer ms mismatches)
    % Use LFP as reference for time dimension; assume others match
    nSamp = size(epoched_data.lfp, 3);
    timeVector = linspace(config.epoch_tframe(1), config.epoch_tframe(2), nSamp); % ms

    %----- Optional interpolation of sharp artifacts inside specified windows
    if isfield(config, 'artifact') && isfield(config.artifact, 'enabled') && config.artifact.enabled
        art = config.artifact;

        % Defaults
        if ~isfield(art, 'windows_ms') || isempty(art.windows_ms)
            warning('config.artifact.enabled=true but no windows specified; skipping interpolation.');
        else
            if ~isfield(art, 'pad_ms') || isempty(art.pad_ms), art.pad_ms = 0; end
            if ~isfield(art, 'method') || isempty(art.method), art.method = 'linear'; end
            if ~isfield(art, 'signals') || isempty(art.signals)
                % Use whatever fields exist in epoched_data
                available = intersect({'lfp','csd','mua'}, fieldnames(epoched_data));
                art.signals = available(:)';
            end

            % Pad windows if requested
            wins = art.windows_ms;
            if size(wins,2) ~= 2, error('artifact.windows_ms must be Nx2 [start end] in ms.'); end
            wins = [wins(:,1)-art.pad_ms, wins(:,2)+art.pad_ms];

            % Apply per-signal
            for s = art.signals
                sig = s{1};
                if ~isfield(epoched_data, sig) || isempty(epoched_data.(sig)), continue; end
                epoched_data.(sig) = interpolate_windows_3d(epoched_data.(sig), timeVector, wins, art.method);
            end
        end
    end

    %----- Define baseline correction period (ms)
    baselineStart = -30;  % Start of baseline window (ms)
    baselineEnd   = -3;   % End of baseline window (ms)

    % Ensure within epoch
    if baselineStart < config.epoch_tframe(1)
        warning('Baseline start is outside the epoch range. Using start of epoch.');
        baselineStart = config.epoch_tframe(1);
    end
    if baselineEnd > config.epoch_tframe(2)
        warning('Baseline end is outside the epoch range. Using end of epoch.');
        baselineEnd = config.epoch_tframe(2);
    end

    % Convert baseline to indices on the actual time vector
    [~, baselineStartIdx] = min(abs(timeVector - baselineStart));
    [~, baselineEndIdx]   = min(abs(timeVector - baselineEnd));
    baselineIdx = baselineStartIdx:baselineEndIdx;

    %% Artifact rejection (median filter) BEFORE baseline correction (kept from original)
    [epoched_data.lfp, ~] = MTF_rejectartifacts(epoched_data.lfp, 'median', 3);
    [epoched_data.csd, ~] = MTF_rejectartifacts(epoched_data.csd, 'median', 3);
    [epoched_data.mua, ~] = MTF_rejectartifacts(epoched_data.mua, 'median', 3);

    %% Baseline Correction (vectorized with implicit expansion; requires R2016b+)
    lfp_bsl = epoched_data.lfp ...
            - mean(epoched_data.lfp(:,:,baselineIdx), 3, 'omitnan');
    csd_bsl = epoched_data.csd ...
            - mean(epoched_data.csd(:,:,baselineIdx), 3, 'omitnan');
    mua_bsl = [];
    if ~isempty(epoched_data.mua)
        mua_bsl = epoched_data.mua ...
                - mean(epoched_data.mua(:,:,baselineIdx), 3, 'omitnan');
    end

    %% Plotting Parameters
    numChannelsLFP = 1:size(lfp_bsl, 1);
    numChannelsCSD = 1:size(csd_bsl, 1);
    if ~isempty(mua_bsl), numChannelsMUA = 1:size(mua_bsl, 1); end

    % Symmetric color limits scaled to 75% of max abs of the mean-across-trials image
    lfp_mean = squeeze(mean(lfp_bsl, 2));
    v = max(abs(lfp_mean(:))); lfp_caxis = 0.75 * [-v v];

    csd_mean = squeeze(mean(csd_bsl, 2));
    v = max(abs(csd_mean(:))); csd_caxis = 0.75 * [-v v];

    if ~isempty(mua_bsl)
        mua_mean = squeeze(mean(mua_bsl, 2));
        v = max(abs(mua_mean(:))); mua_caxis = 0.75 * [-v v];
    end

    %% Plot Figures
    fig = figure('Position', [200, 200, 1200, 700]);

    % LFP
    subplot(1, 3, 1);
    imagesc(timeVector, numChannelsLFP, lfp_mean);
    title(['Trial Avg. LFP (', condition, ')']);
    xlabel('Time (ms)'); ylabel('Channel');
    colormap(flipud(jet));
    colorbar; caxis(lfp_caxis);

    % CSD
    subplot(1, 3, 2);
    imagesc(timeVector, numChannelsCSD, csd_mean);
    title(['Trial Avg. CSD (', condition, ')']);
    xlabel('Time (ms)'); ylabel('Channel');
    colorbar; caxis(csd_caxis);

    % MUA (optional)
    if ~isempty(mua_bsl)
        subplot(1, 3, 3);
        imagesc(timeVector, numChannelsMUA, mua_mean);
        title(['Trial Avg. MUA (', condition, ')']);
        xlabel('Time (ms)'); ylabel('Channel');
        colorbar; caxis(mua_caxis);
    end

    %% Save
    saveas(fig, fullfile(figuresDir, [fileName, '_', config.trigger_method, condition, '_avg_profiles.fig']));
    saveas(fig, fullfile(figuresDir, [fileName, '_', config.trigger_method, condition, '_avg_profiles.jpg']));
    close(fig);

    disp(['Baseline correction and plots for ', condition, ' have been saved successfully.']);

end

% ========================= Helper =========================
function Xout = interpolate_windows_3d(Xin, t_ms, wins_ms, method)
% Linearly (or pchip) interpolate over specified time windows for a 3D array
% Xin: [channels x trials x time], t_ms: [1 x nTime]
% wins_ms: Nx2 [tStart tEnd] in ms
% method: 'linear' or 'pchip'
    if isempty(Xin), Xout = Xin; return; end
    Xout = Xin;
    nT = numel(t_ms);

    for w = 1:size(wins_ms,1)
        t1 = wins_ms(w,1); t2 = wins_ms(w,2);
        if t2 <= t1, continue; end
        mask = (t_ms >= t1) & (t_ms <= t2);
        if ~any(mask), continue; end

        idx_win = find(mask);
        iL = idx_win(1) - 1;  % left neighbor
        iR = idx_win(end) + 1; % right neighbor

        % Edge cases: if window touches an edge, do nearest-value fill
        if iL < 1 && iR <= nT
            Xout(:,:,idx_win) = repmat(Xout(:,:,iR), 1, 1, numel(idx_win));
            continue;
        elseif iR > nT && iL >= 1
            Xout(:,:,idx_win) = repmat(Xout(:,:,iL), 1, 1, numel(idx_win));
            continue;
        elseif iL < 1 && iR > nT
            % Whole record masked; nothing sensible to do
            continue;
        end

        % Interpolate between neighbors across time for every (chan,trial)
        x = [t_ms(iL) t_ms(iR)];
        xq = t_ms(idx_win);

        % Compute weights for linear interpolation
        %if strcmpi(method, 'linear') || numel(x) < 3
            % weights per query point (row vector)
            wR = (xq - x(1)) ./ (x(2) - x(1));  % 0..1
            wL = 1 - wR;

            % Broadcast across (chan x trial)
            for k = 1:numel(idx_win)
                Xout(:,:,idx_win(k)) = wL(k).*Xout(:,:,iL) + wR(k).*Xout(:,:,iR);
            end
        %else
            % 'pchip' with just two anchor points degenerates to linear;
            % but if you later expand to more anchors, this keeps the API.
            % wR = (xq - x(1)) ./ (x(2) - x(1));  % same as linear
            % wL = 1 - wR;
            % for k = 1:numel(idx_win)
            %     Xout(:,:,idx_win(k)) = wL(k).*Xout(:,:,iL) + wR(k).*Xout(:,:,iR);
            % end
        %end
    end
end


function plot_baseline_corrected_data_mult_trigType(epoched_data, config, figuresDir, fileName)
 % Adapted to handle the new data structure with trigType grouping
disp(['oo you are fancy; Making figures for mult trig types']);

% Define baseline correction period
baselineStart = -30;  
baselineEnd = -3;     

% Ensure baseline period is within the epoch timeframe
if baselineStart < config.epoch_tframe(1)
    warning('Baseline start is outside the epoch range. Using start of epoch.');
    baselineStart = config.epoch_tframe(1);
end
if baselineEnd > config.epoch_tframe(2)
    warning('Baseline end is outside the epoch range. Using end of epoch.');
    baselineEnd = config.epoch_tframe(2);
end

epochTimeframe = config.epoch_tframe(1):config.epoch_tframe(2); 
fs = config.newadrate;

% Find sample indices corresponding to baseline range
[~, baselineStartIdx] = min(abs(epochTimeframe - baselineStart));
[~, baselineEndIdx] = min(abs(epochTimeframe - baselineEnd));
baselineIdx = baselineStartIdx:baselineEndIdx;
durationMs = config.durationForTuningCurve;
[~, durationForTuningCurve] = min(abs(epochTimeframe - durationMs))

% Extract unique trigTypes
uniqueTrigTypes = unique([epoched_data.sortedData.trigType]);

for tt = 1:length(uniqueTrigTypes)
    trigType = uniqueTrigTypes(tt);
    disp(['Processing trigType: ', num2str(trigType)]);

    % Find indices for this trigType
    if ~isnan(trigType)
        indices = find([epoched_data.sortedData.trigType] == trigType);
    else
        
        indices = tt; % for some reason this one broke my brain
    end

    % Collect baseline-corrected data across all indices of this trigType
    lfp_all = [];
    csd_all = [];
    mua_all = [];

    for idx = indices
        data = epoched_data.sortedData(idx);

        % Initialize arrays for this particular entry
        lfp_bsl = zeros(size(data.lfp));
        csd_bsl = zeros(size(data.csd));
        if ~isempty(data.mua)
            mua_bsl = zeros(size(data.mua));
        else
            mua_bsl = [];
        end

        % artifact reject
            [data.lfp, ~] = MTF_rejectartifacts(data.lfp, 'median', 3);
            [data.csd, ~] = MTF_rejectartifacts(data.csd, 'median', 3);
            [data.mua, ~] = MTF_rejectartifacts(data.mua, 'median', 3);


        % Baseline Correction for each channel and trial
        for chct = 1:size(data.lfp, 1)
            for trct = 1:size(data.lfp, 2)
                % LFP Baseline Correction
                lfp_bsl(chct, trct, :) = squeeze(data.lfp(chct, trct, :)) - ...
                    mean(squeeze(data.lfp(chct, trct, baselineIdx)), 'omitnan');
            end
        end
        % Baseline Correction for each channel and trial
        for chct = 1:size(data.csd, 1)
            for trct = 1:size(data.csd, 2)
                % CSD Baseline Correction
                csd_bsl(chct, trct, :) = squeeze(data.csd(chct, trct, :)) - ...
                    mean(squeeze(data.csd(chct, trct, baselineIdx)), 'omitnan');
            end
        end
                % Baseline Correction for each channel and trial
        for chct = 1:size(data.mua, 1)
            for trct = 1:size(data.mua, 2)

                % MUA Baseline Correction (if available)
                if ~isempty(mua_bsl)
                    mua_bsl(chct, trct, :) = squeeze(data.mua(chct, trct, :)) - ...
                        mean(squeeze(data.mua(chct, trct, baselineIdx)), 'omitnan');
                end
            end
        end

        % Concatenate the baseline-corrected data
        lfp_all = cat(2, lfp_all, lfp_bsl);
        csd_all = cat(2, csd_all, csd_bsl);
        if ~isempty(mua_bsl)
            mua_all = cat(2, mua_all, mua_bsl);
        end
    end

    % Trial averaging across the second dimension (trials)
    lfp_avg = mean(lfp_all, 2, 'omitnan');
    csd_avg = mean(csd_all, 2, 'omitnan');
    if ~isempty(mua_all)
        mua_avg = mean(mua_all, 2, 'omitnan');
    else
        mua_avg = [];
    end

    % Plotting parameters
    timeVector = linspace(config.epoch_tframe(1), config.epoch_tframe(2), size(lfp_avg, 3));
    numChannels = 1:size(lfp_avg, 1);
    numChannelsCSD = 1:size(csd_avg, 1);

    % Determine symmetrical color axis
    lfp_min = min(lfp_avg, [], 'all');
    lfp_max = max(lfp_avg, [], 'all');
    lfp_caxis = [-max(abs([lfp_min, lfp_max])), max(abs([lfp_min, lfp_max]))];

    csd_min = min(csd_avg, [], 'all');
    csd_max = max(csd_avg, [], 'all');
    csd_caxis = [-max(abs([csd_min, csd_max])), max(abs([csd_min, csd_max]))];

    if ~isempty(mua_avg)
        mua_min = min(mua_avg, [], 'all');
        mua_max = max(mua_avg, [], 'all');
        mua_caxis = [-max(abs([mua_min, mua_max])), max(abs([mua_min, mua_max]))];
    end

    % Plot Figures
    fig = figure('Position', [200, 200, 1200, 700]);

    % Plot LFP
    subplot(1, 3, 1);
    imagesc(timeVector, numChannels, squeeze(lfp_avg));
    title(['Trial Avg. LFP (' ', TrigType ', num2str(trigType), ')']);
    xlabel('Time (ms)');
    ylabel('Channel');
    colormap(flipud(jet));
    colorbar;
    caxis(lfp_caxis);

    % Plot CSD
    subplot(1, 3, 2);
    imagesc(timeVector, numChannelsCSD, squeeze(csd_avg));
    title(['Trial Avg. CSD (' ', TrigType ', num2str(trigType), ')']);
    xlabel('Time (ms)');
    ylabel('Channel');
    colorbar;
    caxis(csd_caxis);

    % Plot MUA (if available)
    if ~isempty(mua_avg)
        subplot(1, 3, 3);
        imagesc(timeVector, numChannelsCSD, squeeze(mua_avg));
        title(['Trial Avg. MUA (' ', TrigType ', num2str(trigType), ')']);
        xlabel('Time (ms)');
        ylabel('Channel');
        colorbar;
        caxis(mua_caxis);
    end

    % Save Figures
    saveas(fig, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType', num2str(trigType), '_avg_profiles.fig']));
    saveas(fig, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType', num2str(trigType), '_avg_profiles.jpg']));
    close(fig);

    disp(['Baseline correction and plots for trigType ', num2str(trigType), ' have been saved successfully.']);

    % save mean
    
    if ~isempty(mua_avg)
        mua_avg = squeeze(mua_avg);
        tuningcurve(tt,:) = mean(mua_avg(:,baselineEndIdx:durationForTuningCurve),2);
    end
    
    
end

% Generate average MUA data matrix (e.g., tuningcurve)
% Assume tuningcurve is of size [numTrigTypes, numChannels]
% Where rows = trigger types (e.g., tone frequencies), columns = channels
for norm_count = 1:size(tuningcurve,2)
    tuningcurve(:,norm_count) = tuningcurve(:,norm_count)/max(tuningcurve(:,norm_count));
end


numTrigTypes = size(tuningcurve, 1);
numChannels = size(tuningcurve, 2);


% Plot MUA Mean for Each Channel as a Function of TrigType
figMUA = figure('Position', [200, 200, 1200, 700]);

hold on;
% Determine a dynamic offset that avoids overlap
channelRanges = max(tuningcurve, [], 1) - min(tuningcurve, [], 1);  % range per channel
%offset = max(channelRanges) * 1.5;  % buffer to avoid overlaps
offset = 0.4;

colors = lines(size(tuningcurve, 2));  % Generate unique colors for each channel
numChannels = size(tuningcurve, 2);

% Generate YData to mimic imagesc behavior: Channel 1 at top, Channel N at bottom
yData = fliplr(1:numChannels);  % Reversing the order of channels

% Plot each channel, but assign them to a reversed y-axis order
for ch = 1:numChannels
    % Calculate the offset for each channel in reversed order
    y_offset = find(yData == ch) * offset;  % Find where the current channel would be plotted on the flipped y-axis
    plot(1:length(uniqueTrigTypes), tuningcurve(:,ch) + y_offset, 'Color', colors(ch,:),'LineWidth',1);
end

% Add labels and title
set(gca, 'LineWidth', 1.5);  %thicccc
set(gca, 'YDir', 'normal');  
%set(gca, 'XScale', 'log')

title('Mean MUA');
%xlabel('Tone Freq (kHz)');
%xlabel('Mod Freq')
ylabel('Channels (Offset for Clarity)');
set(gca,'FontSize',16)

% Set Y-ticks to match the desired order (Ch 1 at the top, Ch N at the bottom)
yticks((1:numChannels) * offset);
yticklabels(arrayfun(@(x) sprintf('Ch %d', x), yData, 'UniformOutput', false));

xticks(1:length(uniqueTrigTypes));  % Make sure xticks match your trigger types

if nanmean(uniqueTrigTypes) > 300
    xticklabels({'0.3','0.5','0.7','1','1.4','2','2.8','4','5.6','8','11','16','22','32','Noise'});
end
    
grid on;
hold off;

% Save Figures
saveas(figMUA, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType_MUA_TuningCurve.fig']));
saveas(figMUA, fullfile(figuresDir, [fileName, '_', config.trigger_method, '_trigType_MUA_TuningCurve.jpg']));
end



function rawData = remap_channels(rawData)
    % REMAP_CHANNELS - Adjusts raw data channels to correct hardware mismatch
    %
    % Parameters:
    % rawData: Original raw electrophysiology data matrix (channels x trials x samples)
    %
    % Output:
    % rawData: Remapped raw data with corrected channel alignment

    fprintf('Applying channel remapping ...\n');
    X = 1:2:size(rawData, 1) - 1; % Odd indices
    X1 = 2:2:size(rawData, 1);    % Even indices

    % Swap channels
    remappedData = rawData; % Copy original data
    remappedData(X, :,:) = rawData(X1, :,:); % Odd -> Even
    remappedData(X1, :,:) = rawData(X, :,:); % Even -> Odd

    rawData = remappedData;
    
end


function [rawChannels, analogChannels, digitalChannels] = classify_channels(hFile)
    % Classify channels as raw or analog based on labels
    rawChannels = [];
    analogChannels = [];
    numEntities = length(hFile.Entity);
    digitalChannels = [];

    for i = 1:numEntities
        label = hFile.Entity(i).Label;
        if contains(label, 'raw', 'IgnoreCase', true)
            rawChannels(end + 1) = i;
        elseif contains(label, 'analog', 'IgnoreCase', true)
            analogChannels(end + 1) = i;
        elseif contains(hFile.Entity(i).EntityType, 'Event', 'IgnoreCase', true)
            digitalChannels(end + 1) = i;
        end
    end
end

function data = get_channel_data(hFile, channels)
    % GET_CHANNEL_DATA - Retrieve data for specified channels sequentially
    %
    % Parameters:
    % hFile: Handle to the Neuroshare file
    % channels: Array of channel IDs to import
    %
    % Output:
    % data: Matrix (channels x samples) of retrieved analog data
    
    data = [];
    for ch = channels
        [~, ~, channelData] = ns_GetAnalogData(hFile, ch, 1, 1e8);
        data = [data; channelData']; %#ok<AGROW> Transpose for consistency
    end
end





function [triggers_standard, triggers_deviant] = get_analog_triggers(hFile, analogChannel, threshold)
    % GET_ANALOG_TRIGGERS - Detect first pulse of each pulse train in analog data
    %
    % Parameters:
    % hFile          - Handle to the Neuroshare file
    % analogChannel  - Analog channel ID
    % threshold      - Threshold for detecting rising edges
    %
    % Outputs:
    % triggers_standard - Indices of the first pulse of standard trains (~5 pulses)
    % triggers_deviant  - Indices of the first pulse of deviant trains (~10 pulses)
    
    %% Step 1: Retrieve Analog Data
    [ns_RESULT, ~, analogData] = ns_GetAnalogData(hFile, analogChannel, 1, hFile.Entity(analogChannel).Count);

    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Error retrieving analog data: %s', ns_RESULT);
    end
    
    
    % check sampling rate
for check = 1:length(hFile.FileInfo)
    if isfield(hFile.FileInfo(check), 'FileTypeID') && ...
       isfield(hFile.FileInfo(check), 'TimeSpan') && ...
       isfield(hFile.FileInfo(check), 'Label')
   
        % Display File Info
        fprintf('File Type: %s | Time Span: %.2f  | Sampling Rate: %s\n', ...
            hFile.FileInfo(check).FileTypeID, ...
            hFile.FileInfo(check).TimeSpan, ...
            hFile.FileInfo(check).Label);
        
        % Validate Sampling Rate if TimeSpan is valid
        if hFile.FileInfo(check).TimeSpan > 0
            if contains(hFile.FileInfo(check).Label, '30 ksamp/sec')
                fprintf('Found 30 kHz samp rate! The bar for success is low.\n');
            else
                warning('Check yourself before you wreck yourself, samp rate may not be 30k (Found: %s).', ...
                    hFile.FileInfo(check).Label);
            end
        else
            fprintf('Skipping entry %d: TimeSpan is zero or invalid.\n', check);
        end
    else
        warning('Incomplete metadata in FileInfo entry %d. Missing FileTypeID, TimeSpan, or Label.', check);
    end
end




    %% Step 2: Detect All Rising Edges
    allTriggers = find(diff(analogData > threshold) == 1); % Rising edge detection
    if length(allTriggers) < 10
        disp('WARNING: less than 10 triggers')
    end

    %% Step 3: Separate Pulse Trains
    fs = 30000;
    minPulseGap = 0.15 * fs; % Minimum gap to separate pulse trains
    pulseTrainStarts = [];
    pulseTrainEnds = [];
    lastPulse = -inf; % Track the last pulse's index

    % Identify train start and end points
    for i = 1:length(allTriggers)
        if allTriggers(i) - lastPulse > minPulseGap
            % New train detected
            pulseTrainStarts(end + 1) = allTriggers(i); %#ok<AGROW>
            if i > 1
                pulseTrainEnds(end + 1) = allTriggers(i - 1); %#ok<AGROW>
            end
        end
        lastPulse = allTriggers(i);
    end
    % Add final train end
    pulseTrainEnds(end + 1) = allTriggers(end);

    %% Step 4: Classify Trains as Standard or Deviant
    triggers_standard = [];
    triggers_deviant = [];

    for i = 1:length(pulseTrainStarts)
        % Extract pulses within this train
        startIdx = find(allTriggers >= pulseTrainStarts(i), 1, 'first');
        endIdx = find(allTriggers <= pulseTrainEnds(i), 1, 'last');
        trainPulses = allTriggers(startIdx:endIdx);

        % Count number of pulses in the train
        pulseCount = length(trainPulses);

        if pulseCount >= 8 % Deviant trains (~10 pulses)
            triggers_deviant(end + 1) = pulseTrainStarts(i); %#ok<AGROW>
        elseif pulseCount >= 4 && pulseCount <= 7 % Standard trains (~5 pulses)
            triggers_standard(end + 1) = pulseTrainStarts(i); %#ok<AGROW>
        end
    end

    %% Step 5: Optional Plot for Visualization
%     figure;
%     plot(1:length(analogData), analogData);
%     hold on;
%     plot(triggers_standard, analogData(triggers_standard), 'go', 'MarkerSize', 8, 'LineWidth', 2);
%     plot(triggers_deviant, analogData(triggers_deviant), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
%     title('Analog Signal with Detected Standard and Deviant Triggers');
%     xlabel('Sample Index');
%     ylabel('Analog Signal');
%     legend('Analog Signal', 'Standard Triggers (~5 pulses)', 'Deviant Triggers (~10 pulses)');
%     hold off;
% 
%     disp(['Detected ', num2str(length(triggers_standard)), ' standard pulse trains and ', num2str(length(triggers_deviant)), ' deviant pulse trains.']);
end


function [triggers_standard] = get_segment_triggers(hFile, analogChannel, threshold)
    % GET_ANALOG_TRIGGERS - Detect first pulse of each pulse train in analog data
    % repurposed get analog triggers function to find stimulation artifact
    % to use as a trigger (forgot to record Stim in ripple)
    % Parameters:
    % hFile          - Handle to the Neuroshare file
    % analogChannel  - Analog channel ID
    % threshold      - Threshold for detecting rising edges
    %
    % Outputs:
    % triggers_standard - Indices of pulses exceeding threshold
    % 
    
    %% Step 1: Retrieve Analog Data
    [ns_RESULT, ~, analogData] = ns_GetAnalogData(hFile, analogChannel, 1, hFile.Entity(analogChannel).Count);

    if ~strcmp(ns_RESULT, 'ns_OK')
        error('Error retrieving analog data: %s', ns_RESULT);
    end
    
    
    % check sampling rate
for check = 1:length(hFile.FileInfo)
    if isfield(hFile.FileInfo(check), 'FileTypeID') && ...
       isfield(hFile.FileInfo(check), 'TimeSpan') && ...
       isfield(hFile.FileInfo(check), 'Label')
   
        % Display File Info
        fprintf('File Type: %s | Time Span: %.2f  | Sampling Rate: %s\n', ...
            hFile.FileInfo(check).FileTypeID, ...
            hFile.FileInfo(check).TimeSpan, ...
            hFile.FileInfo(check).Label);
        
        % Validate Sampling Rate if TimeSpan is valid
        if hFile.FileInfo(check).TimeSpan > 0
            if contains(hFile.FileInfo(check).Label, '30 ksamp/sec')
                fprintf('Found 30 kHz samp rate! The bar for success is low.\n');
            else
                warning('Check yourself before you wreck yourself, samp rate may not be 30k (Found: %s).', ...
                    hFile.FileInfo(check).Label);
            end
        else
            fprintf('Skipping entry %d: TimeSpan is zero or invalid.\n', check);
        end
    else
        warning('Incomplete metadata in FileInfo entry %d. Missing FileTypeID, TimeSpan, or Label.', check);
    end
end




    %% Step 2: Detect All Rising Edges
    allTriggers = find(diff(analogData > threshold) == 1); % Rising edge detection
    if length(allTriggers) < 10
        disp('WARNING: less than 10 triggers')
    end

    %% Step 3: Separate Pulse Trains
    fs = 30000;
    minPulseGap = 0.5 * fs; % Minimum gap to separate pulse trains
    pulseTrainStarts = [];
    
    lastPulse = -inf; % Track the last pulse's index

    % Identify train start and end points
    for i = 1:length(allTriggers)
        if allTriggers(i) - lastPulse > minPulseGap
            % New pulse detected
            pulseTrainStarts(end + 1) = allTriggers(i); %#ok<AGROW>

        end
        lastPulse = allTriggers(i);
    end

triggers_standard = pulseTrainStarts;


    %% Step 4 Optional Plot for Visualization
    figure;
    plot(1:length(analogData), analogData);
    hold on;
    plot(triggers_standard, analogData(triggers_standard), 'go', 'MarkerSize', 8, 'LineWidth', 2);
    
    title('Analog Signal with Detected Standard and Deviant Triggers');
    xlabel('Sample Index');
    ylabel('Analog Signal');
    
    hold off;

    disp(['Detected ', num2str(length(triggers_standard)), '  pulses trains ']);
end


function timingResults = AV40_importEyelink(dataDirectory, filename)
    % AV40_IMPORT_EYELINK - Import and parse TIMING_RESULTS and DEVIANT_RESPONSE_RESULTS
    %
    % Inputs:
    %   dataDirectory: Path to the main data dir containing edf subdir
    %   filename: Name of the current data file (e.g., 'pt027000029.nev')
    %
    % Output:
    %   timingResults: Struct containing extracted event times for specified events
    
    %% Step 1: Locate the 'edf' Subdirectory
    edfDir = fullfile(dataDirectory, 'edf');
    if ~isfolder(edfDir)
        error('The ''edf'' subdirectory does not exist in: %s', dataDirectory);
    end
    
    % Remove file extension
    [~, baseFilename, ~] = fileparts(filename);
    %fprintf('Searching for subdirectory: %s in %s\n', baseFilename, edfDir);

    %% Step 2: Locate Matching Subdirectory
    subdirs = dir(edfDir);
    subdirs = subdirs([subdirs.isdir]); % Keep only directories
    matchingSubdir = '';
    
    for i = 1:length(subdirs)
        if strcmp(subdirs(i).name, baseFilename)
            matchingSubdir = fullfile(edfDir, subdirs(i).name);
            break;
        end
    end
    
    if isempty(matchingSubdir)
        error('No subdirectory matching "%s" was found in: %s', baseFilename, edfDir);
    end

    %% Step 3: Locate TIMING_RESULTS and DEVIANT_RESPONSE_RESULTS
    timingFilePath = fullfile(matchingSubdir, 'TIMING_RESULTS.txt');
    deviantFilePath = fullfile(matchingSubdir, 'DEVIANT_RESPONSE_RESULTS.txt');
    
    if ~isfile(timingFilePath)
        error('TIMING_RESULTS.txt not found in: %s', matchingSubdir);
    end
    if ~isfile(deviantFilePath)
        warning('DEVIANT_RESPONSE_RESULTS.txt not found in: %s', matchingSubdir);
        deviantFilePath = ''; % Allow proceeding without deviant file
    end
    
    %% Step 4: Parse TIMING_RESULTS
    fprintf('Parsing TIMING_RESULTS from: %s\n', timingFilePath);
    timingResults = struct();
    timingResults.ADDT_STANDARD = [];
    timingResults.ADDT_DEVIANT = [];
    timingResults.VDDT_STANDARD = [];
    timingResults.VDDT_DEVIANT = [];
    timingResults.VST_ONSET = [];
    
    fid = fopen(timingFilePath, 'rt');
    rawLines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    rawLines = rawLines{1};

    % Extract headers
    headers = strsplit(rawLines{1}, '\t');
    eventTimeIdx = find(strcmp(headers, 'EVENT_TIME'));

    if isempty(eventTimeIdx)
        warning('EVENT_TIME column not found in TIMING_RESULTS.txt');
    end

    % Process data lines
    for i = 2:length(rawLines)  % Start from second line (skip headers)
        line = strtrim(rawLines{i});
        if contains(line, 'ADDT_STANDARD_ONSET')
            timingResults.ADDT_STANDARD(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'ADDT_DEVIANT_ONSET')
            timingResults.ADDT_DEVIANT(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'VDDT_STANDARD_ONSET')
            timingResults.VDDT_STANDARD(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'VDDT_DEVIANT_ONSET')
            timingResults.VDDT_DEVIANT(end + 1) = extract_event_time(line, eventTimeIdx);
        elseif contains(line, 'VST_ONSET')
            timingResults.VST_ONSET(end + 1) = extract_event_time(line, eventTimeIdx);
        end
    end
    
    %% Step 5: Parse DEVIANT_RESPONSE_RESULTS (if available)
    if ~isempty(deviantFilePath)
        fprintf('Parsing DEVIANT_RESPONSE_RESULTS from: %s\n', deviantFilePath);
        timingResults.DEVIANT_HIT_REWARD_JUICE_ONSET = [];
        timingResults.FALSE_ALARM_BUTTON_RESPONSE = [];
        timingResults.DEVIANT_MISS = [];
        
        fid = fopen(deviantFilePath, 'rt');
        rawLines = textscan(fid, '%s', 'Delimiter', '\n');
        fclose(fid);
        rawLines = rawLines{1};

        % Extract headers
        headers = strsplit(rawLines{1}, '\t');
        eventTimeIdx = find(strcmp(headers, 'EVENT_TIME'));

        if isempty(eventTimeIdx)
            warning('EVENT_TIME column not found in DEVIANT_RESPONSE_RESULTS.txt');
        end

        % Process data lines
        for i = 2:length(rawLines)
            line = strtrim(rawLines{i});
            if contains(line, 'DEVIANT_HIT_REWARD_JUICE_ONSET')
                timingResults.DEVIANT_HIT_REWARD_JUICE_ONSET(end + 1) = extract_event_time(line, eventTimeIdx);
            elseif contains(line, 'FALSE_ALARM_BUTTON_RESPONSE')
                timingResults.FALSE_ALARM_BUTTON_RESPONSE(end + 1) = extract_event_time(line, eventTimeIdx);
            elseif contains(line, 'DEVIANT_MISS')
                timingResults.DEVIANT_MISS(end + 1) = extract_event_time(line, eventTimeIdx);
            end
        end
    else
        warning('Skipping DEVIANT_RESPONSE_RESULTS parsing (file missing).');
    end

    %% Helper Function to Extract Event Time
    function eventTime = extract_event_time(line, eventTimeIdx)
        % Extract EVENT_TIME using the correct column index
        tokens = strsplit(line, '\t');
        eventTime = NaN;

        if isempty(eventTimeIdx)
            warning('EVENT_TIME column not detected, returning NaN.');
            return;
        end
        
        if length(tokens) >= eventTimeIdx
            eventTime = str2double(tokens{eventTimeIdx});
            if isnan(eventTime)
                warning('Failed to extract valid EVENT_TIME from line: %s', line);
            end
        else
            warning('EVENT_TIME column index exceeds available columns in line: %s', line);
        end
    end
end







function config = get_default_config()
    % Define default processing parameters
    config.epoch_tframe = [-50, 150]; % Epoch window in ms
    config.newadrate = 1000;          % Resampling rate
    config.filters.lfp = [0.5, 300];  % LFP filter range (Hz)
    config.filters.mua = [300, 5000]; % MUA filter range (Hz)
    config.trigger_channel = 93;  % Default analog trigger channel
    config.channels = [];             % Auto-detect raw channels
    config.trigger_method = 'analog'; % 'events' or 'analog'
    config.trigger_threshold = 50;   % Threshold for analog trigger detection
    config.event_entity_id = 1;       % Default Event Entity ID
    config.artifact_threshold = 3;    % Z-score threshold for artifact rejection
    config.checksync = 0;
    config.get_deviant = 0;
end
