function [FEMesh, cutOutElements] = cutoutMesh2_arb_pick(Nstrips, testt,maxMeshSize,minMeshSize,outerRadiusmax,diagnostic)

% innerRadius = inner radius of the cutout circle
% outerRadius = inner radius of the cutout circle
% Nstrips = number of strips/cuts
% dTheta = radial angle of each cut
% maxMeshSize = max size of an element
% minMeshSize = min size of an element
% Additional points to capture the circular nature of the circle

gMatLength =50;% 1 + 1 + numel(ptPoly_x) * 2;

gd = zeros(gMatLength, Nstrips+1);

ptss=zeros(Nstrips,gMatLength);
ptssbound=zeros(Nstrips,2);

%% draw a circle
c = Nstrips
gd(1:4, c) = [1,0,0,outerRadiusmax]'; %% This is for circle
gd(1:10, 2) = [2,4,0,outerRadiusmax,0,-outerRadiusmax,outerRadiusmax,0,-outerRadiusmax,0]'; %% This is for circle
dnn=3;
gd(1:10, 3) = [2,4,0,outerRadiusmax/dnn,0,-outerRadiusmax/dnn,outerRadiusmax/dnn,0,-outerRadiusmax/dnn,0]'; %% This is for circle
dnn=2;
gd(1:10, 4) = [2,4,0,outerRadiusmax/dnn,0,-outerRadiusmax/dnn,outerRadiusmax/dnn,0,-outerRadiusmax/dnn,0]'; %% This is for circle

size(gd);
stringss(c)='c';
%%
g = decsg(gd);
% de
model = createpde;

geometryFromEdges(model,g);
% Plot the geometry (for simple check)
if diagnostic>0
%     figure(1);
    pdegplot(model,'EdgeLabels','off')
%     axis equal
%     xlim([-1.1,1.1])
end
%  return
if diagnostic<2
    %% Mesh it
    FEMesh = generateMesh(model,'Hmax', 2*minMeshSize, 'GeometricOrder', 'linear');% 'Hmin', ...
    nodes2D = FEMesh.Nodes;
    numm=Nstrips;
%     circlePartNo=[1,6,7]; %% for circle
    circlePartNo=1:7;%2:numm+1;
    circlePartNo;
    cutOutElements = findElements(FEMesh,'region','Face', circlePartNo);
    
    figure(2)
    % F = pdegplot(model, 'FaceLabels','on');
    pdeplot(model);
    hold on
%     plot(nodes2D(1,:), nodes2D(2,:), 'ko');
    cutOutElements = findElements(FEMesh,'region','Face',circlePartNo);
    pdemesh(FEMesh.Nodes, FEMesh.Elements(:,cutOutElements),'EdgeColor','green');
    hold off
    axis equal
    pp=1.1;

    disp('nnn')
    size(FEMesh.Nodes)
    size(FEMesh.Elements)
    size(FEMesh.Elements(:,cutOutElements))

    
end
return
