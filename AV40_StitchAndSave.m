% stitch data and save
% chase m 2025

clear;clc;close all
path = '/Volumes/Samsung03/data/AV40/Peter/pt035/imported/Vis20kHz/pt035000016_analog_continuous.mat';
outputDir = '/Volumes/Samsung03/data/AV40/Peter/pt035/imported/Vis20kHz/';
load(path);

%% get inputs for stitching continuous data
% Get size of data, our AV40 import script stores continuous data as 1
% second chunks, so the number of chunks indicates the length of the
% recording, and the chunk duration is the number of samples in each, thus
% the sampling frequency (fs) is equal to the chunk duration (samps in 1s)
[numChannels,numChunks,chunkDuration] = size(continuous_raw);
desiredDuration = numChunks;
fs = chunkDuration;
tic 
[stitchedData, triggers_std_analog_trimmed] = AV40_StitchContinuousData(continuous_raw, triggers_std_analog, desiredDuration, fs);
toc
disp('stitched data done')

%% HP filter for units
craw.cnt = stitchedData;
craw.adrate = chunkDuration;
craw.arej = [];
tic
[~, ~, ~, ~, cntu, ~] = module_cnt05(craw, 20000, [0.5 300], [300 5000], 1);
toc

%% Save the continuous HP filtered data
save(fullfile(outputDir, 'cntu_workspace.mat'), 'cntu', 'triggers_std_analog_trimmed','config', '-v7.3');
save(fullfile(outputDir,'cntu_pt035000016.mat'), 'cntu','config', '-v7.3');

disp('unit data saved!')