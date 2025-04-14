% stitch data and save

clear;clc;close all
path = '/Volumes/Samsung03/data/AV40/Peter/pt034/imported/Vsearch20kHz/pt034000025_analog_continuous.mat';
outputDir = '/Volumes/Samsung03/data/AV40/Peter/pt034/imported/Vsearch20kHz/';
load(path);

[numChannels,numChunks,chunkDuration] = size(continuous_raw);
desiredDuration = numChunks;
fs = chunkDuration;
tic 
[stitchedData, triggers_std_analog_trimmed] = AV40_StitchContinuousData(continuous_raw, triggers_std_analog, desiredDuration, fs);
toc
disp('stitched data done')
% filter for units
craw.cnt = stitchedData;
craw.adrate = chunkDuration;
craw.arej = [];
tic
[~, ~, ~, ~, cntu, ~] = module_cnt05(craw, 20000, [0.5 300], [300 3000], 1);
toc

save(fullfile(outputDir, 'cntu_workspace.mat'), 'cntu', 'triggers_std_analog_trimmed','config', '-v7.3');
save(fullfile(outputDir,'cntu_pt034000025.mat'), 'cntu','config', '-v7.3');

disp('unit data saved!')