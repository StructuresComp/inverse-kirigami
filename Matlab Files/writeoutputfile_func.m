function [ok] = writeoutputfile_func(num,testt, R)
    train=0;
    plott=1;
    limval=[];
    flipp=1;
    datasvdall=[];
    checkcoordinatedir=1;
    energyall=[];
    outputenergy=1;
    outerRadius = R;
    E = 1 ;
    mu = 0.5;
    Nstripsall=[];
    testnumall=[];ZZall1=[];ZZall2=[];
    loc = '';
    screensize = importdata('screensize.mat');
    screensize = double(screensize)
    screensize = screensize*1.0%*1.1%*1.2  %% for 923 sens plots
    for iii=1%%:length(Nstripsall0)
    %     iii
    if train==1
        Nstripsall=1:13;
    else
        Nstripsall=1;
    end
    for Nstrips=Nstripsall%Nstripsall0(iii)%10:13
        if train==1
            out=importdata(['C:\Users\leixi\OneDrive\Desktop\sci03\',['verttest2',num2str(Nstrips)],'.mat']);
        % testnumall0(iii)%
            C=out.C;
            length(C)
        else
            C=1;
        end      
        for testnum = C
            if train==1
                jobName= ['mvertvertshape-vae_',num2str(Nstrips),'_',num2str(testnum)];
                outputDataall=[loc,jobName,'_outputall.txt'];
                savefigall = [loc,jobName,'_figout.png'];
                savematall = [loc,jobName,'_figout.mat'];
            else
                jobName= ['mvertr-j-alpha-try'];
                outputDataall=[loc,jobName,'_outputall.txt'];
                savefigall = [loc,'mvertr-test',num2str(testt),'num',num2str(num),'_figout.png'];
                savematall = [loc,'mvertr-test',num2str(testt),'num',num2str(num),'_figout.mat'];

            end
    
            if isfile(outputDataall) == true 
                datadis = load(outputDataall);
                if length(datadis)>0
                    numm=size(datadis );
                    xxnode=datadis (1:numm/3);
                    yynode=datadis (1+numm/3:numm/3*2);
                    zznode=datadis (numm/3*2+1:end);
                    maxxx=max(xxnode);
                    minxx=min(xxnode);
                    maxyy=max(yynode);
                    minyy=min(yynode);
                    maxzz=max(zznode);
                    minzz=min(zznode);
                    pp=1.1;
                    ptmax=find(zznode==maxzz);
                    ptmin=find(zznode==minzz);
                    difff=500;
                    dshape =64*4;
                    x0 =linspace(-R*screensize,R*screensize,dshape);
                    y0 =linspace(-R*screensize,R*screensize,dshape);
                    [x1, y1] =meshgrid(x0,y0) ;
                    [fitresult,gof]=createFit(xxnode,yynode,zznode);
    
                    rr=(0.02:0.02:1)*R;
                    count=1;
                    for theta = (0:0.02:1)*2*pi
                            xii=rr*cos(theta);yii=rr*sin(theta);
                            zii = fitresult(xii, yii);
                            rmax(count)=max(sqrt(xii.^2+yii.^2).*abs(zii)./abs(zii));
                            count = count+1;
                    end
                    maxrr = max(rmax);
                    minrr = min(rmax); 


                    z1 = fitresult(x1, y1);
                    if plott==1
                        figure('units','normalized','outerposition',[0 0 1 1])%('units','pixels','outerposition',[0 0 1920        1080])%
                        hold on
                        
                        z1mod =abs(z1-min(min(zznode)));
                        z1mod(isnan(z1mod)>0)=nan;                        
                        surf(x1,y1, z1mod,'FaceColor','interp', 'EdgeColor', 'none');
    
                        pp=1.1;view(2);
                        xlim([-pp*outerRadius*screensize,pp*outerRadius*screensize]);        ylim([-pp*outerRadius*screensize,pp*outerRadius*screensize]);
                        colormap('jet')
                        axis equal
                        
                        caxis([0,R*screensize]);
                             
                        grid off
                        axis off
                        set(gcf,'color','w');
                        nsize=2;
                        set(gcf,'PaperUnits','inches', 'PaperSize', [nsize,nsize],'PaperPosition',[0 0 nsize nsize])
                        saveas(gcf,savefigall)

                        save(savematall, 'z1mod')
                        ok=1;
                    end
    
                    close all;
                else
                end                
     
            end
     
        end
    end
    %%
    end
