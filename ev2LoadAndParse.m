function [ev2Table,selectedColumn] = ev2LoadAndParse(path,varname)

    % loads and parses ev2 file given a file path
    
    ev2Path = "C:\MultStim\TrialParameters\ot037038025_2025-03-26_12-48-33.ev2"
    ev2Table = readtable(ev2Path, 'FileType', 'text', 'Delimiter', '\t');
    
    % pull out the headers
    [headers] = ev2Table.Properties.VariableNames;
    
    % check them for matches to our varname
    for headcount = 1:length(headers)
        if contains(headers(headcount),varname)
            selectedColumn = ev2Table.(headcount);
        end
    end


end