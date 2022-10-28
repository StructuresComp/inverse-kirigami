function write_input_file_arb(newInputFile, nodes1File,nodes2File, nodes3File, assemblyFile, bcfile, trilayer,prestretch)
%
% This function simply reads the Kirigami_v6.inp file and replaces the
% names of the necessary files by the random string-based file names
% generated by the MATLAB codes.
%

originalFile = 'Kirigami_v6.inp';

% Read txt into cell A
fid = fopen(originalFile,'r');
i = 1;
tline = fgetl(fid);
A{i} = tline;

while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    
    k = strfind(tline, 'NodeSubstrate');
    if (numel(k) > 0)
        tline = strrep(tline, 'NodeSubstrate, -0.20', ['NodeSubstrate,  ', num2str(-prestretch)]);
        save prestretch prestretch
    end

    % Replace 'Part1NodesFile.txt' with the given nodes1File
    k = strfind(tline, 'PART1NodesFile.txt');
    if (numel(k) > 0)
        tline = strrep(tline, 'PART1NodesFile.txt',[ nodes1File]);
    end

    % Replace 'Part2NodesFile.txt' with the given nodes2File
    k = strfind(tline, 'PART2NodesFile.txt');
    if (numel(k) > 0)
        tline = strrep(tline, 'PART2NodesFile.txt',[ nodes2File]);
    end

    % Replace 'Part2NodesFile.txt' with the given nodes3File
    if trilayer==1
        k = strfind(tline, 'PART3NodesFile.txt');
        if (numel(k) > 0)
            tline = strrep(tline, 'PART3NodesFile.txt', [nodes3File]);
        end
    else
        k = strfind(tline, 'PART3NodesFile.txt');
        if (numel(k) > 0)
            tline = strrep(tline, 'PART3NodesFile.txt', '*');
        end
    end

    % Replace 'Part2NodesFile.txt' with the given assemblyFile
    k = strfind(tline, 'assemblyMATLAB.txt');
    if (numel(k) > 0)
        tline = strrep(tline, 'assemblyMATLAB.txt', assemblyFile);
    end
    
    % Replace 'bcMATLAB.txt' with the given bcfile
    k = strfind(tline, 'bcMATLAB.txt');
    if (numel(k) > 0)
        tline = strrep(tline, 'bcMATLAB.txt', bcfile);
    end
    
    A{i} = tline;
end
fclose(fid);

% Write cell A into txt
fid = fopen(newInputFile, 'w');
for i = 1:numel(A)
    if A{i+1} == -1
        fprintf(fid,'%s', A{i});
        break
    else
        fprintf(fid,'%s\n', A{i});
    end
end

end
