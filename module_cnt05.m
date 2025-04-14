function [cnt_arej, cnte, cntm, cntc, cntu, cntb] = module_cnt05(craw, newadrate, filtere, filteru, filtertype);

% filtere = [0.5 300];
% filteru = [300 0];

adrate = craw.adrate;

cnte = zeros(size(craw.cnt));
cntu = zeros(size(craw.cnt));

if filteru(2)>adrate/2
    filteru(2)=adrate/2.01;
end

if filtertype == 2
    
    bpFilt = designfilt('bandpassfir','FilterOrder',200, ...
        'CutoffFrequency1',filtere(1),'CutoffFrequency2',filtere(2), ...
        'SampleRate',adrate);
    
    % filtering to get field
    
    for i=1:size(craw.cnt,1)
        cnte(i,:)=filtfilt(bpFilt,craw.cnt(i,:));
    end
    
    bpFilt = designfilt('bandpassfir','FilterOrder',200, ...
        'CutoffFrequency1',filteru(1),'CutoffFrequency2',filteru(2), ...
        'SampleRate',adrate);
    
    for i=1:size(craw.cnt,1)
        cntu(i,:)=filtfilt(bpFilt,craw.cnt(i,:));
    end
    
    
    
else

    % filtering to get field
    if filtere(1)==0
        n = 2;
        Wn = filtere(2)/(adrate/2);
        [b,a] = butter(n,Wn,'low');
        for i=1:size(craw.cnt,1)
            cnte(i,:)=filtfilt(b,a,craw.cnt(i,:));
        end
    else
        n = 2;
        Wn = filtere/(adrate/2);
        [b,a] = butter(n,Wn);
        for i=1:size(craw.cnt,1)
            cnte(i,:)=filtfilt(b,a,craw.cnt(i,:));
        end
    end
    
    %%%% filtering to get the unit/MUA
    if filteru(2)==0
        n = 2;
        Wn = filteru(1)/(adrate/2);
        [b,a] = butter(n,Wn,'high');
        for i=1:size(craw.cnt,1)
            cntu(i,:)=filtfilt(b,a,craw.cnt(i,:));
        end
    else
        n = 2;
        Wn = filteru/(adrate/2);
        [b,a] = butter(n,Wn);
        for i=1:size(craw.cnt,1)
            cntu(i,:)=filtfilt(b,a,craw.cnt(i,:));
        end
    end
    
end



cntu    = cntu(2:end-1,:);



cntm = zeros(size(cntu));
% rectifying unit
for i=1:size(cntu,1)
    cntm(i,:)=abs(hilbert(cntu(i,:)));
end




if adrate ~= newadrate
    
    % filtering before downsample
    nyq = newadrate/2.01;
    
    if filtertype == 1
        
        
        n = 2;
        Wn = nyq/(adrate/2);
        [b,a] = butter(n,Wn,'low');
        for i=1:size(cnte,1)
            cnte(i,:)=filtfilt(b,a,cnte(i,:));
            
            
        end
        for i=1:size(cntu,1)
            
            cntm(i,:)=filtfilt(b,a,cntm(i,:));
            
            cntu(i,:)=filtfilt(b,a,cntu(i,:));
            
        end
        
    else
        
        bpFilt = designfilt('lowpassfir','FilterOrder',200, ...
            'CutoffFrequency',nyq, ...
            'SampleRate',adrate);
        for i=1:size(cnte,1)
            cnte(i,:)=filtfilt(bpFilt,cnte(i,:));
            
            
        end
        for i=1:size(cntu,1)
            
            cntm(i,:)=filtfilt(bpFilt,cntm(i,:));
            
            cntu(i,:)=filtfilt(bpFilt,cntu(i,:));
            
        end
    end
    
    
    % downsample or resmaple
    downsampleby    = adrate/newadrate;
    if downsampleby-round(downsampleby)==0
        cnte            = downsample(cnte',downsampleby)';
        
        cntm            = downsample(cntm',downsampleby)';
        cntu            = downsample(cntu',downsampleby)';
        
        try
            cnt_arej        = downsample(craw.arej',downsampleby)';
        catch
            cnt_arej        = [];
        end
    else
        cnte            = resample(cnte',newadrate,adrate)';
        
        cntm            = resample(cntm',newadrate,adrate)';
        cntu            = resample(cntu',newadrate,adrate)';
        
        try
            cnt_arej        = resample(craw.arej',newadrate,adrate)';
        catch
            cnt_arej        = [];
        end
    end
    
else
    
    try
        size(cnt_arej);
    catch
        cnt_arej        = [];
    end
end


cntc    = -diff(cnte,2,1); %CSD
cntb    = diff(cnte,1,1); %bipolar derivation 
cntb    = cntb(1:end-1,:); %bipolar derivation minus one channel to conveniently match CSD
cnte    = cnte(2:end-1,:); % raw(?) monopolar(?) LFP

