

clear all; close all;

directory1='/Users/chase/Desktop/NKI/data/VisDevDet/240103_2/';
directory2='/Users/chase/Desktop/NKI/data/VisDevDet/240103_2/';

%filename='231228_3.asc';
filename='240103_2.asc';


% Open ASC file & load all lines

    [fid, message] = fopen([directory1 filename],'rt');%open eyelink File
    % [fid, message] = fopen(fullfile(currFile.elDir,currFile.asc{1}),'rt');%open eyelink File
       
    if fid == -1 disp(message); end % display message if file can't be opened

    I=1;
    while feof(fid) == 0
             E.EB_Events{I,1} = fgetl(fid); %store messages
             I = I+1; %took out the semicolor at the end of the line so that can see which line it's up to
    end
    fclose(fid)
    clear fid I message



% Go through each line and extract the information you want
    
    E.n_EB_Events = length(E.EB_Events);
    trialID=[];
    n_START=0;
    ALL_trial_timing=[];
    ALL_trial_incorrect=[];
    ALL_trial_correct=[];

     for edfLine = 1 : E.n_EB_Events %parse through all lines
        str = strtrim(E.EB_Events{edfLine}); %get string on current line
        
        if findstr(str,'TRIALID') % what we are currently using as trial time
        n_START = n_START + 1;
        m_trialTiming=[];
        m_incorrect=[];
        m_correct=[];   
    
        i_space = findstr(str,' ');
        v_trialTiming(:,1) = str2num(str(4:i_space(1)));
        m_trialTiming(size(m_trialTiming,1)+1,:) = v_trialTiming;
        ALL_trial_timing(n_START,1)=m_trialTiming;
        ALL_trial_timing(n_START,2)=edfLine;
        end
    
        if findstr(str,'blackScreenTimeOutTimer_5sec') % indicates incorrect trial
        i_space = findstr(str,' ');
        v_incorrect(:,1) = str2num(str(4:i_space(1)));
        m_incorrect(size(m_incorrect,1)+1,:) = v_incorrect;
        ALL_trial_incorrect(n_START,1)=m_incorrect;
        ALL_trial_incorrect(n_START,2)=edfLine;
        end
    
    
        if findstr(str,'deviantDetection') % indicates correct trial
        i_space = findstr(str,' ');
        v_correct(:,1) = str2num(str(4:i_space(1)));
        m_correct(size(m_correct,1)+1,:) = v_correct;
        ALL_trial_correct(n_START,1)=m_correct;
        ALL_trial_correct(n_START,2)=edfLine;
        end
    
     end


% Create 1 variable that lists the TRIALID time, the edfLine, and if it was
    % behavior was correct (1) or not (0)

     ALL_trials_compiled=[];
     clear i1
     for i1=1:size(ALL_trial_timing,1);
         if i1 <= length(ALL_trial_correct)
             if ALL_trial_correct(i1,1) > 0
                 ALL_trials_compiled(i1,1:2)=ALL_trial_timing(i1,:);
                 ALL_trials_compiled(i1,3)=1;
             end
         end
         if i1 <= length(ALL_trial_incorrect)
             if ALL_trial_incorrect(i1,1) > 0
                 ALL_trials_compiled(i1,1:2)=ALL_trial_timing(i1,:);
                 ALL_trials_compiled(i1,3)=0;
             end
         end
     end


% Set binSize (winSize) for analysis (***ADJUST***) 

    winSize=240000; % in msec 


% Looks at performance over time (with no overlap)

trialsByTime_perf=[];

 clear i1 i2 i3
 i1=ALL_trials_compiled(1,1):winSize:ALL_trials_compiled(end,1)+winSize;

 for i2=2:length(i1);
     A=[];
     A=find(ALL_trials_compiled(:,1) >= (i1(1,i2-1)) & ALL_trials_compiled(:,1) < i1(1,i2));

     if ~isempty(A);
        B=[];B=ALL_trials_compiled(A,:);
        trialsByTime_perf(i2-1,1)=length(find(B(:,3)>0)); % correct trials
        trialsByTime_perf(i2-1,2)=length(find(B(:,3)==0)); % incorrect trials
        trialsByTime_perf(i2-1,3)=length(B(:,3)); % total trials
     else
        trialsByTime_perf(i2-1,1:3)=NaN;
     end
 end

xaxisConversion=[]; minBins=[];
xaxisConversion=winSize/60000;
minBins=xaxisConversion*(1:length(trialsByTime_perf));

minBinsStr={};
clear i1; 
for i1=1:length(minBins);
    minBinsStr{i1,1}=num2str(minBins(1,i1));
end


figure;
subplot(3,1,1)
plot(1:length(trialsByTime_perf),trialsByTime_perf(:,1))
xlim([1 length(trialsByTime_perf)])
xticks(1:5:length(trialsByTime_perf));xticklabels(minBinsStr(1:5:length(minBinsStr)));    
title([ num2str(filename(1:end-4)) ' - Correct Trials Over Time (' num2str(winSize) 'ms bins, no overlap)'])
subplot(3,1,2)
plot(1:length(trialsByTime_perf),trialsByTime_perf(:,2))
xlim([1 length(trialsByTime_perf)])
xticks(1:5:length(trialsByTime_perf));xticklabels(minBinsStr(1:5:length(minBinsStr)));    
title(['Inorrect Trials Over Time'])
ylabel(['count'])
subplot(3,1,3)
plot(1:length(trialsByTime_perf),trialsByTime_perf(:,3))
xlim([1 length(trialsByTime_perf)])
xticks(1:5:length(trialsByTime_perf));xticklabels(minBinsStr(1:5:length(minBinsStr)));    
title(['Total # Trials Over Time'])
xlabel(['minutes'])
print ('-djpeg', '-r300', [directory2 num2str(filename(1:end-4)) ' - perfOverTime_noOverlap_' num2str(winSize) 'msBins.jpg']);


figure;
plot(1:length(trialsByTime_perf),trialsByTime_perf(:,1),1:length(trialsByTime_perf),trialsByTime_perf(:,2),'r',...
    1:length(trialsByTime_perf),trialsByTime_perf(:,3),'g')
legend('correct','incorrect','total')
xlim([1 length(trialsByTime_perf)])
xticks(1:5:length(trialsByTime_perf));xticklabels(minBinsStr(1:5:length(minBinsStr)));    
title([ num2str(filename(1:end-4)) ' - Performance Over Time (' num2str(winSize) 'ms bins, no overlap)'])
ylabel('count')
xlabel(['minutes'])
print ('-djpeg', '-r300', [directory2 num2str(filename(1:end-4)) ' - perfOverTime_noOverlap_' num2str(winSize) 'msBins_2.jpg']);




% Looks at performance over time (overlap = 1/2 winSize)

trialsByTime_perf=[];

 clear i1 i2 i3
 i1=ALL_trials_compiled(1,1):winSize/2:ALL_trials_compiled(end,1)+winSize;

 for i2=1:length(i1);
     A=[];
     A=find(ALL_trials_compiled(:,1) >= (i1(1,i2)) & ALL_trials_compiled(:,1) < (i1(1,i2)+winSize));

     if ~isempty(A);
        B=[];B=ALL_trials_compiled(A,:);
        trialsByTime_perf(i2,1)=length(find(B(:,3)>0)); % correct trials
        trialsByTime_perf(i2,2)=length(find(B(:,3)==0)); % incorrect trials
        trialsByTime_perf(i2,3)=length(B(:,3)); % total trials
     else
        trialsByTime_perf(i2,1:3)=NaN;
     end
 end

 xaxisConversion=[]; minBins=[];
 xaxisConversion=winSize/60000;
 minBins=xaxisConversion*(1:length(trialsByTime_perf));
 

figure;
subplot(3,1,1)

% Remove NaN values from x and y
x = (1:length(trialsByTime_perf))';
y = trialsByTime_perf(:,1);

nanIndices = isnan(y);

x = x(~nanIndices);
y = y(~nanIndices);

plot(x, y, 'LineWidth', 2)
xlim([1 length(trialsByTime_perf)])
title([ num2str(filename(1:end-4)) ' - Correct Trials Over Time (' num2str(winSize) 'ms bins, no overlap)'])


% Fit a sine wave
fitSine = fit(x, y, 'sin1');

% Plot the sine wave
hold on;
plot(fitSine, 'r--', 'LineWidth', 2);
legend('Correct Trials', 'Fitted Sine Wave');
hold off;

subplot(3,1,2)
plot(1:length(trialsByTime_perf),trialsByTime_perf(:,2))
xlim([1 length(trialsByTime_perf)])
title(['Incorrect Trials Over Time'])
ylabel({'count'})
subplot(3,1,3)
plot(1:length(trialsByTime_perf),trialsByTime_perf(:,3))
xlim([1 length(trialsByTime_perf)])
title(['Total # Trials Over Time'])
xlabel([num2str(xaxisConversion) ' minute bins'])
print ('-djpeg', '-r300', [directory2 num2str(filename(1:end-4)) ' - perfOverTime_halfOverlap_' num2str(winSize) 'msBins.jpg']);


figure;
plot(1:length(trialsByTime_perf),trialsByTime_perf(:,1),1:length(trialsByTime_perf),trialsByTime_perf(:,2),'r',...
    1:length(trialsByTime_perf),trialsByTime_perf(:,3),'g')
legend('correct','incorrect','total')
xlim([1 length(trialsByTime_perf)])
title([ num2str(filename(1:end-4)) ' - Performance Over Time (' num2str(winSize) 'ms bins, half overlap)'])
ylabel({'count'})
xlabel([num2str(xaxisConversion) ' minute bins'])
print ('-djpeg', '-r300', [directory2 num2str(filename(1:end-4)) ' - perfOverTime_halfOverlap_' num2str(winSize) 'msBins_2.jpg']);





