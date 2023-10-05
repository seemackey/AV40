% testing software trigger
    clear;
    close all;
    % set up actx server/control 
    handles.RP = actxcontrol('RPco.x');
    RP	= handles.RP;

    % connect to the device and halt any ongoing processes
    RP.ConnectRP2('USB',1);
    RP.Halt;
    RP.ClearCOF;
    
    % load our rcx file and run it
    %RP.LoadCOF('C:\TDT\RPvdsEx\Examples\TDT_test_soft_trig.rcx');
    RP.LoadCOF('C:\Users\seema\OneDrive\Documents\MATLAB\ytdt_tono_tictoc.rcx');
    RP.Run; %
    
    %% set params
    
    RP.SetTagVal('ampclickstd',0.5);
    
    
    disp(RP.GetTagVal('ReadAmp')) % checking
    RP.SetTagVal('ampclickdev',0);
    %RP.SetTagVal('amp',1);
    %RP.SetTagVal('frq',1000);
    %disp(RP.GetTagVal('ReadMe')) % checking
    %RP.SetTagVal('dur',0.1);
    
    %% boom goes the dynamite. trig 2 is for clicks, trig 1 for other program
    RP.SoftTrg(2)
    
    %invoke(RP,'halt');