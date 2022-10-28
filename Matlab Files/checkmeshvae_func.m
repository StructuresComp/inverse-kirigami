function [noelkir,iiall]=checkmeshvae_func(xscaleall, yscaleall,zall,R,ssize);
    
    a= importdata('r-nodes1-full.inp');
    a=a.data;
    b= importdata('r-nodes2-full.inp');
    b=b.data;
    ael=importdata('elementall-full.txt');
    for ii = 1:length(ael)
        ely(ii)=mean(a(ael(ii,2:4),2));
        elx(ii)=mean(a(ael(ii,2:4),3));
    
    end
    elxscale=(elx+R)/(2*R);
    elyscale=(ely+R)/(2*R);
    iiall=[];
    count=1;
    
    vall=griddata(xscaleall, yscaleall,zall,elxscale,elyscale);
    for ii = 1:length(ael)
        if vall(ii)>80  %% To determine kirigami 
            iiall=[iiall;[count, ael(ii,2:end)]];
            count=count+1;
        end
    end
    noelkir = count-1;



%% need to convert image to coordinate to element numbers



% % a= importdata('r-nodes1_12_1.inp').data;
% % b= importdata('r-nodes2_12_1.inp').data;
% % figure
% % plot(a(:,2),a(:,3),'o')
% % ael=importdata('elementall.txt');
% % for ii = 1:length(ael)
% %     elx(ii)=mean(a(ael(ii,2:4),2));
% %     ely(ii)=mean(a(ael(ii,2:4),3));
% % 
% % end
% % figure
% % plot(elx,ely,'o')
% % 
