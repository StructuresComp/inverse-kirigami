function [k] = meshGen_shell_v1_arb_pick_wsize (outerRadius, ...
    thickness_1, thickness_2, ...
    Nstrips,testt, ...
    maxMeshSize, minMeshSize, ...
    inputName, nodes1File, nodes2File, nodes3File, assemblyFile, bcfile,trilayer,diagnostic,prestretch,ssize)
%
%% Create an input (.inp) file
write_input_file_arb(inputName, nodes1File, nodes2File, nodes3File, assemblyFile, bcfile, trilayer,prestretch);


[FEMesh, cutOutElements] = cutoutMesh2_arb_pick(Nstrips,...
testt, maxMeshSize, minMeshSize,outerRadius,diagnostic);

%%
nodes2D = FEMesh.Nodes*ssize;
%save nodes2Dnew nodes2D;
nodes2D= importdata('nodes2Dnew.mat')*ssize;

x2D = nodes2D(1, :);
y2D = nodes2D(2, :);
num_nodes2D = length(x2D);
xedge=x2D';
yedge=y2D';
k = boundary(xedge,yedge);
el2D = FEMesh.Elements;
%save el2Dnew el2D;
el2D= importdata('el2Dnew.mat');

length(nodes2D)
length(el2D)

% return
[~,num_el2D] = size(el2D);
%%
fid1 = fopen(nodes1File, 'w'); %% first file
fid2 = fopen(nodes2File, 'w'); %% second file
fid4 = fopen('elestry.inp','w'); %% second fileu 
%% Nodes
fprintf(fid1, '*Part, name =Substrate\n');
fprintf(fid1, '*Node\n');

fprintf(fid2, '*Part, name =Top Kirigami\n');
fprintf(fid2,'*Node\n');


if trilayer==1
    fid3 = fopen(nodes3File, 'w'); %% third file
    fprintf(fid3, '*Part, name =Bottom Kirigami\n');
    fprintf(fid3,'*Node\n');
end

for c2D = 1:num_nodes2D
    nodeNo = c2D;
    % z = (cHeight-1) * thickness; % z = 0 for shell element
    fprintf(fid1, '%d, %f, %f, %f\n', nodeNo, x2D(c2D), y2D(c2D), 0.0);
    fprintf(fid2, '%d, %f, %f, %f\n', nodeNo, x2D(c2D), y2D(c2D), 0.0);
    if trilayer==1
        fprintf(fid3, '%d, %f, %f, %f\n', nodeNo, x2D(c2D), y2D(c2D), 0.0);
    end
end
%% Part1 Elements of Substrate

fprintf(fid1, '*Element, type=S3R\n'); %% 3-node triangular shell element
for c = 1:num_el2D
        fprintf(fid1, '%d, %d, %d, %d\n', c, el2D(1, c),el2D(2, c), ...
            el2D(3, c));
end
%% Part2 Elements of Top Kirigami layer

[num_el_kir_top,el_Kir_top]= tryreadimage_func(outerRadius,ssize);

size(el_Kir_top)

% return
fprintf(fid2, '*Element, type=S3R\n'); %% 3-node triangular shell element

for c = 1:num_el_kir_top
        fprintf(fid2, '%d, %d, %d, %d\n', c, el_Kir_top(c,2),el_Kir_top(c,3), ...
            el_Kir_top(c,4));
end

fprintf(fid4, '*Element, type=S3R\n'); %% 3-node triangular shell element

for c = 1:num_el_kir_top
        fprintf(fid4, '%d, %d, %d, %d\n', c, el_Kir_top(c,2),el_Kir_top(c,3), ...
            el_Kir_top(c,4));
end

%% Part3 Elements of Bottom Kirigami layer
if trilayer==1
    num_el_kir_bottom = numel(cutOutElements); 
    el_Kir_bottom = zeros(3, num_el_kir_bottom);
    for c2D = 1:num_el_kir_bottom
        cEl = cutOutElements(c2D);
        elNo = c2D ;  % element number
        
        el_Kir_bottom( 1, elNo) = el2D(1, cEl) ;
        el_Kir_bottom( 2, elNo) = el2D(2, cEl) ;
        el_Kir_bottom( 3, elNo) = el2D(3, cEl) ;
        
    end

    fprintf(fid3, '*Element, type=S3R\n'); %% 3-node triangular shell element

    for c = 1:num_el_kir_bottom
            fprintf(fid3, '%d, %d, %d, %d\n', c, el_Kir_bottom(1, c),el_Kir_bottom(2, c), ...
                el_Kir_bottom(3, c));
    end
end

%% Part1 Element and Node sets
% Shell layer
fprintf(fid1,'*Nset, nset = NodeSubstrate, generate\n'); % all nodes in the substrate layer
fprintf(fid1,'%d, %d, 1\n', 1, num_nodes2D);

fprintf(fid1, '*Elset, elset=elSubstrate, generate\n'); % all elements in the substrate layer
fprintf(fid1, '%d, %d, 1\n', 1, num_el2D);

% Substrate_contact_Surface_top
fprintf(fid1, '*Elset, elset = Substrate_contact_surface_top_SPOS, internal, generate\n');
fprintf(fid1,'%d, %d, 1\n',1, num_el2D);
fprintf(fid1,'*Surface, type=ELEMENT, name=Substrate_contact_surface_top\n');
fprintf(fid1,'Substrate_contact_surface_top_SPOS, SPOS\n');
fprintf(fid1,'**Section: Section membrane\n');
fprintf(fid1,'*Shell Section, elset=elSubstrate, material=Material-Substrate, offset=SPOS\n');
fprintf(fid1,'%f, %d\n', thickness_1, 5);

% Substrate_contact_Surface_bottom
if trilayer ==1
    fprintf(fid1, '*Elset, elset = Substrate_contact_surface_bottom_SNEG, internal, generate\n');
    fprintf(fid1,'%d, %d, 1\n',1, num_el2D);
    fprintf(fid1,'*Surface, type=ELEMENT, name=Substrate_contact_surface_bottom\n');
    fprintf(fid1,'Substrate_contact_surface_bottom_SNEG, SNEG\n');
    fprintf(fid1,'**Section: Section membrane\n');
    fprintf(fid1,'*Shell Section, elset=elSubstrate, material=Material-Substrate, offset=SNEG\n');
    fprintf(fid1,'%f, %d\n', thickness_1, 5);
end

%  Center Node
% Find node located at (0,0,0)
xDesired = 0.0;
yDesired = 0.0;
dist = sqrt( (x2D - xDesired).^2 + (y2D - yDesired).^2 );
[~, minInd] = min(dist);
% % minInd
fprintf(fid1, '*Nset, nset=centerNode\n%d\n', minInd);
fprintf(fid1,'*End Part\n');

fclose(fid1);

%% Part2 Element and Node sets
% Top Kirigami layer

fprintf(fid2, '*Elset, elset=elKir_top, generate\n');
fprintf(fid2, '%d, %d, 1\n', 1, num_el_kir_top);

% membrane_contact_Surface
fprintf(fid2, '*Elset, elset = Top_Kirigami_contact_surface_SNEG, internal, generate\n');
fprintf(fid2,'%d, %d, 1\n', 1, num_el_kir_top);
fprintf(fid2,'*Surface, type=ELEMENT, name=Top_Kirigami_contact_surface\n');
fprintf(fid2,'Top_Kirigami_contact_surface_SNEG, SNEG\n');
fprintf(fid2,'**Section: Section membrane\n');
fprintf(fid2,'*Shell Section, elset=elKir_top, material=Material-Kirigami, offset=SNEG\n');
fprintf(fid2,'%f, %d\n', thickness_2, 5);
fprintf(fid2,'*End Part\n');

fclose(fid2);

%% Part3 Element and Node sets
% Bottom Kirigami layer
if trilayer==1

    fprintf(fid3, '*Elset, elset=elKir_bottom, generate\n');
    fprintf(fid3, '%d, %d, 1\n', 1, num_el_kir_bottom);

    % membrane_contact_Surface
    fprintf(fid3, '*Elset, elset = Bottom_Kirigami_contact_surface_SPOS, internal, generate\n');
    fprintf(fid3,'%d, %d, 1\n', 1, num_el_kir_bottom);
    fprintf(fid3,'*Surface, type=ELEMENT, name=Bottom_Kirigami_contact_surface\n');
    fprintf(fid3,'Bottom_Kirigami_contact_surface_SPOS, SPOS\n');
    fprintf(fid3,'**Section: Section membrane\n');
    fprintf(fid3,'*Shell Section, elset=elKir_bottom, material=Material-Kirigami, offset=SPOS\n');
    fprintf(fid3,'%f, %d\n', thickness_2, 5);
    fprintf(fid3,'*End Part\n');

    fclose(fid3);
end

%% Assembly File
fid = fopen(assemblyFile,'w'); % Assembly file
fprintf(fid,'*Assembly, name = Assembly\n');
fprintf(fid, '*Instance,name=substrate-layer, part = Substrate\n*End Instance\n');
fprintf(fid,'*Instance,name= top-kirigami-layer, part = Top Kirigami\n*End Instance\n');
if trilayer==1
    fprintf(fid,'*Instance,name= bottom-kirigami-layer, part = Bottom Kirigami\n*End Instance\n');
end
fprintf(fid,'*Nset, nset = NodeSubstrate, instance = substrate-layer, generate\n');
fprintf(fid,'%d, %d, 1\n', 1, num_nodes2D);
fprintf(fid, '*Elset, elset=elSubstrate, instance = substrate-layer, generate\n');
fprintf(fid, '%d, %d, 1\n', 1, num_el2D);
fprintf(fid, '*Nset, nset=centerNode, instance = substrate-layer\n');
fprintf(fid, '%d,\n', minInd);
fprintf(fid, '*Elset, elset=elKir_top,instance = top-kirigami-layer, generate\n');
fprintf(fid, '%d, %d, 1\n', 1, num_el_kir_top);
if trilayer==1
    fprintf(fid, '*Elset, elset=elKir_bottom,instance = bottom-kirigami-layer, generate\n');
    fprintf(fid, '%d, %d, 1\n', 1, num_el_kir_bottom);
end
% Substrate Contact Surface
fprintf(fid, '*Elset, elset = Substrate_contact_surface_top_SPOS, instance = substrate-layer, internal, generate\n');
fprintf(fid,'%d, %d, 1\n',1, num_el2D);
if trilayer==1
    fprintf(fid, '*Elset, elset = Substrate_contact_surface_bottom_SNEG, instance = substrate-layer, internal, generate\n');
    fprintf(fid,'%d, %d, 1\n',1, num_el2D);
end
% Top and Bottom Kirigami Contact Surface
fprintf(fid, '*Elset, elset = Top_Kirigami_contact_surface_SNEG, instance = top-kirigami-layer, internal, generate\n');
fprintf(fid,'%d, %d, 1\n',1, num_el_kir_top);
if trilayer==1
    fprintf(fid, '*Elset, elset = Bottom_Kirigami_contact_surface_SPOS, instance = bottom-kirigami-layer, internal, generate\n');
    fprintf(fid,'%d, %d, 1\n',1, num_el_kir_bottom);
end
fprintf(fid,'*Nset, nset=T-CYLINDRICAL_CSYS, instance = substrate-layer, internal\n');
fprintf(fid,'NodeSubstrate\n');
fprintf(fid,'*Transform, nset=T-CYLINDRICAL_CSYS, type=C\n');
fprintf(fid,'%d, %d, %d, %d, %d, %d\n',0.0, 0.0, 0.0, 0.0, 0.0, 1.);
fprintf(fid,'** Constraint: Constraint-1\n');
fprintf(fid,'*Tie, name=Constraint-1, adjust=yes\n');
fprintf(fid,'top-kirigami-layer.Top_Kirigami_contact_surface, substrate-layer.Substrate_contact_surface_top\n');
if trilayer==1    
    fprintf(fid,'** Constraint: Constraint-2\n');
    fprintf(fid,'*Tie, name=Constraint-2, adjust=yes\n');
    fprintf(fid,'bottom-kirigami-layer.Bottom_Kirigami_contact_surface, substrate-layer.Substrate_contact_surface_bottom\n');
end

fprintf(fid,'*End Assembly\n');
fprintf(fid,'*Amplitude, name = amp-1\n');
fprintf(fid,'%f, %f, %f, %f\n', 0.0 , 0.0, 1.0, 1.0);
fclose(fid);

%% Boundary Conditions and Stresses

fid = fopen(bcfile,'w');

fprintf(fid,'**Name: BC1 Type: Displacement/Rotation\n');
fprintf(fid,'*Boundary\n');
fprintf(fid,'Substrate-layer.centerNode, 1, 6, %f\n',0.0);

fclose(fid4);
