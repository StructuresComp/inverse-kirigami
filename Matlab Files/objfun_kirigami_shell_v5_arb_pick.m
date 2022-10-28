function [quality, energy, height, temperature] = objfun_kirigami_shell_v5_arb_pick(inputs, testt)

%% Extract the input data
outerRadius = inputs(1)
thickness_1 = inputs(2);
thickness_2 = inputs(3);
Nstrips = inputs(4);

maxMeshSize = inputs(7);
minMeshSize = inputs(8);
diagnostic = inputs(9);
platform = inputs(10);
trilayer = inputs(11)
prestretch = inputs(12)
length(inputs)
if length(inputs)>12
    ssize = inputs(13)
else
    ssize = 1
end
%%
critTime = 1*3600; % If the Abaqus simulation does not finish within
% "critTime" (seconds), it is forcefully closed

checkInt = 30; % Check status of Abaqus simulation every few seconds 

%% STEP 1: Generate mesh

% Name of the job
jobName = sprintf( 'r-j-alpha-try');

% Name of the input file. It uses Kirigami_v2.inp (or some other "base"
% input file as the foundation and then copies+edits it. 
% See write_input_file() function for details.
inputName =sprintf( 'r-j-alpha-try.inp');

% The input file requies FOUR more files that will need to be written by
% MATLAB:
% (1) nodes1File (nodes and elements of substrate layer)
% (2) nodes2File (nodes and elements of top Kirigami layer)
% (3) assemblyFile (builds the model and applies constraints)
% (4) bcfile (contains boundary conditions)
% (5) nodes3File (nodes and elements of bottom Kirigami layer)

% Name of the file containing the nodes and elements of the substrate layer
nodes1File = sprintf( 'r-nodes1.inp');

% Name of the file containing the nodes and elements of the top Kirigami layer
nodes2File = sprintf( 'r-nodes2.inp');

eles2File = sprintf( 'r-eles2.inp');

% Name of the file containing the nodes and elements of the bottom Kirigami layer
nodes3File = sprintf( 'r-nodes3.inp');

% Assembly file
assemblyFile = sprintf( 'r-assembly.inp');

% Boundary condition file
bcfile = sprintf( 'r-boundaryCondition.inp');
%%
% Run the function that generates the files
if length(inputs)<=12

    k = meshGen_shell_v1_arb_pick(outerRadius, ...
        thickness_1, thickness_2, ...
        Nstrips, testt, ...
        maxMeshSize, minMeshSize, ...
        inputName, nodes1File, nodes2File,nodes3File, assemblyFile, bcfile, trilayer,diagnostic,prestretch);
else
    k = meshGen_shell_v1_arb_pick_wsize(outerRadius, ...
        thickness_1, thickness_2, ...
        Nstrips, testt, ...
        maxMeshSize, minMeshSize, ...
        inputName, nodes1File, nodes2File,nodes3File, assemblyFile, bcfile, trilayer,diagnostic,prestretch,ssize);
end
%%
if platform == 1
    Cmd1 = ['/home/quantum2/abaqusCAE/abaqus j=', jobName, ' input=', inputName, ' &'];
    Cmd2 = ['/home/quantum2/abaqusCAE/abaqus terminate job=', jobName];
%     Cmd3 = ['/home/quantum2/abaqusCAE/abaqus cae noGUI=readODB_v4 -- ', odbname];
elseif platform == 4
    Cmd1 = ['/home/sci02/abaqus/Commands/abaqus j=', jobName, ' input=', inputName, ' &'];
    Cmd2 = ['/home/sci02/abaqus/Commands/abaqus terminate job=', jobName];
%     Cmd3 = ['/home/khalidjm/abaqusInstallation/Commands/abaqus cae noGUI=readODB_v4 -- ', odbname];
end
%% STEP 2: Run simulation
% fprintf('Abaqus start running\n');

% Create the abaqus command and run it
%  Cmd = ['/home/khalidjm/abaqus/Commands/abaqus j=', jobName, ' input=', inputName, ' &'];
% Cmd = ['/home/quantum2/abaqusCAE/abaqus j=', jobName, ' input=', inputName, ' &'];
system(Cmd1);
pause(5);
%%
% Something to keep in mind:
% For some reason, the abaqus software on the desktop does not close by
% itself. The remaining of this code section deals with this problem. It
% checks the status using .sta file every few seconds (variable checkInt).
% If this was not problem, we would simply need the following two commands
% instead of this entire section.
% Cmd = ['/home/khalidjm/abaqus/Commands/abaqus j=', jobName, ' input=', inputName];
% system(Cmd);


% Check for *sta file
staFile = [jobName, '.sta'];

jobComplete = false;
odbname = [jobName, '.odb'];

ind = [];

runTime = 0;
while (jobComplete == false)
    
    % If file doesn't exist, let us continue
    if isfile(staFile) == false
        continue
    end
    
    fid = fopen(staFile, 'r');
    while ~feof(fid)
        tline = fgetl(fid);
        % disp(tline)
    end
    fclose(fid);
    
    ind = strfind(tline, 'HAS COMPLETED SUCCESSFULLY');
    ind2 = strfind(tline, 'HAS NOT BEEN COMPLETED');
%     if numel(ind)>0 || numel(ind2)>0 || runTime > critTime
%         jobComplete = true;

if numel(ind)>0
   jobComplete = true;
   fprintf('THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n');
%    system(['/home/quantum2/abaqusCAE/abaqus terminate job=', jobName]);
   system(Cmd2);
%   system(['fuser -k ', jobName, '*']);
 system(['pkill -9 standard'])
elseif numel(ind2)>0
    jobComplete = true;
    fprintf('THE ANALYSIS HAS NOT BEEN COMPLETED\n');
    system(Cmd2);
%       system(['fuser -k ', jobName, '*']);
   system(['pkill -9 standard'])

elseif runTime > critTime
    jobComplete = true;
        %Kill standard
       % system(['C:\SIMULIA\Commands\abaqus.bat terminate job=', jobName]);
%         system(['/home/quantum2/abaqusCAE/abaqus terminate job=', jobName]);
    system(Cmd2);
   system(['pkill -9 standard'])
        
    else
        jobComplete = false;
    end
    
    pause(checkInt);
    runTime = runTime + checkInt;
end
%
% At this point, if "ind" is empty, the simulation did not finish
% successfully. Otherwise, if numel(ind)>0, the simulation finished
% successfully.
%
if platform == 1
  
    Cmd3 = ['/home/quantum2/abaqusCAE/abaqus cae noGUI=readODB_v72 -- ', odbname];
elseif platform == 2

    Cmd3 = ['/home/lin/abaqus/Commands/abaqus cae noGUI=readODB_v72 -- ', odbname];
elseif platform == 3
   
    Cmd3 = ['/home/khalidjm/abaqusInstallation/Commands/abaqus cae noGUI=readODB_v72 -- ', odbname];
elseif platform == 4
    Cmd3 = ['/home/sci02/abaqus/Commands/abaqus cae noGUI=readODB_v72 -- ', odbname];
elseif platform == 5
    Cmd3 = ['abaqus cae noGUI=readODB_v72 -- ', jobName,'.odb'];
    Cmd4 = ['abaqus cae noGUI=readODB_v8 -- ', jobName,'.odb'];
    
end

%if numel(ind)>0 % successful completion
%% STEP 3: Read ODB file
if isfile(odbname) == true
    system(Cmd3);
else
    fprintf('%s file does not exist\n', odbname);
end

%% STEP 4: Check quality of sphere
jobName ='mvertr-j-alpha-try';
outputData = [jobName,'_output.txt'];
data = load(outputData);
energy = data(:,1);
height = data(:,2);
temperature = data(:,3);
quality = 1;


%% Delete files
if diagnostic < 0.5 % if true, do not delete files
    system('rm abaqus*');
    system(['rm ', jobName, '*']);
    system(['rm ', inputName, ' ', nodes1File, ' ', nodes2File, ' ', nodes3File, ' ',...
        assemblyFile, ' ', bcfile]);
    % Quick command to delete all files: rm as-* n1-* n2-* b-* rand-*
end

system(['rm r-boundaryCondition.inp']);
system(['rm r-j-alpha-try.inp']);
system(['rm r-assembly.inp']);
system(['rm r-nodes1.inp']);
system(['rm r-nodes2.inp']);
system('rm r-j-alpha-try.lck');
system('rm r-j-alpha-try.stt');


system('rm r-j-alpha-try*');


end
