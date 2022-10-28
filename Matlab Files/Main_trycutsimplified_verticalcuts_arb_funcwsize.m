function ok=Main_trycutsimplified_verticalcuts_arb_funcwsize(ssize, prestretch)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%    this code is for the kirigami simulation    %%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    close all;
    clc;
    R = 2/100;  % Original outerRadius in meter
    testt=6; %% test==4  is cross shape, test==5 is full circle test==6 is arbb
    screensize = importdata('screensize.mat')
    screensize = double(screensize)

    ssize= ssize*screensize
    if testt==3
    
        %%% 4 points for each rectangle 
        dataall=importdata('flowchart0.xlsx');
        Nstrips = size(dataall,1)/4;
        Xk=dataall(:,1);
        Yk=dataall(:,2);
    elseif testt==4
        H=R/10*1.25;
        dataall=importdata('flowchart0.xlsx');
        Nstrips=1;
        %% vertical cuts
        Xk=H/2;
        Yk=sqrt(R^2-Xk^2);    
    elseif testt>=5
        H=R/10*1.25*0.001;
        dataall=importdata('flowchart0.xlsx');
        Nstrips=1;
        %% vertical cuts
        Xk=H/2;
        Yk=sqrt(R^2-Xk^2);    
    end
    
    
    locs='';
    %% Inputs parameters
    alphaall=1;
    radiusRatioall = 1; % ratio of the inner Radius to the Outer Radius
    for Nstrips=Nstrips;
    
    % poolObj = parpool(4);
    for testnumcount0=1
    % %     testnum=testnumcount;
    %     Nstrips=4;
        flipp=0;
    % %     cd 'C:\Users\leixi\Box\allodb' %% where the abaqus is saved
        close all;
        testnum=1;
    
        Nstrips_input=Nstrips;
    %     Nstrips_input=dataall(1,testnum);
        alpha_input=1;
        radiusRatio_input=1; % ratio of the inner Radius to the Outer Radius
        stretch_input=1;
        outerRadiusKir_input=1;
        R = 2/100;  % Original outerRadius in meter
        outerRadius=R;
        outerRadiuskir =outerRadiusKir_input*R;  % Original outerRadius in meter
        % alpha = 0.160506329113924; % N_c* Dtheta
        alpha = alpha_input;%%0.34*2
        Nstrips = Nstrips_input; % NUmber of cut is big enough
        radiusRatio=radiusRatio_input;
        stretch=stretch_input;
        
        dTheta = 2*pi*alpha/Nstrips;                
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        platform = 4;  % this is for the path to run the abaqus calculator. The same usage as the circle code
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %% material properties
        C1 = 4.8391; % [kPa] 
        C2 = 14.536;% [kPa]    
        mu = 2 * (C1 + C2); % Shear modulus for linear case (small strains) G
        v = 0.49; % Poisson's ratio
        E = 2 * (1 + v) * mu; % Elastic Modulus for linear case
        
        thickness_1 = 1.0 / 1000; % Thickness of the substrate
        thickness_2 = 1.4/ 1000; % Thickness of the kirigami
        
        maxMeshSize = R/20; % max size of mesh
        minMeshSize = maxMeshSize/2; % min size of mesh
        
        diagnostic = 1; % if diagnostic=1, no files will be deleted and a plot
        
        %% Inputs & container
        if testt<3
            inputs = [radiusRatio, R,  thickness_1, thickness_2, ...
                Nstrips, dTheta,maxMeshSize, minMeshSize, diagnostic, ...
                platform,alpha,outerRadiuskir,symmetric,stretch,testnumcount,flipp];
        %     return
            [~, ENERGY, HEIGHT, TEMPERATURE] = objfun_kirigami_shell_v5_vert(inputs,y1_start,y1_end,y0_start,y0_end,x1,x0);
        elseif testt==4
            inputs = [R, thickness_1, thickness_2, ...
                Nstrips, 0, 0,maxMeshSize, minMeshSize, diagnostic, ...
                platform, 0];
            
            [~, ENERGY, HEIGHT, TEMPERATURE] = objfun_kirigami_shell_v5_arb(inputs, Xk, Yk);
        else
            inputs = [R, thickness_1, thickness_2, ...
                Nstrips, 0, 0,maxMeshSize, minMeshSize, diagnostic, ...
                platform, 0, prestretch, ssize];
            
            [~, ENERGY, HEIGHT, TEMPERATURE] = objfun_kirigami_shell_v5_arb_pick(inputs, testt);
            
        end
            % objfun_kirigami_shell_v5(inputs);
    
        num=1;
        ok= writeoutputfile_func(num,testt, R)
                
    end
    end