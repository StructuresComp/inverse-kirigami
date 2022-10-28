function [noelkir,iiall]= tryreadimage_func(outerRadius,ssize);
readfull=0;
loc='';
if readfull==1
    data0=imread([loc,'samplefull.png']);
    data00= rgb2gray(data0);
    data0=data00(1:10:end,1:10:end);
    [xall0,yall0]=find(data0>100);
    
    xmin=min(-xall0);
    ymin=min(yall0);
    xmax=max(-xall0);
    ymax=max(yall0);
    
    all=[xmin,xmax,ymin,ymax];
    save all all
end
% return
all=importdata([loc,'all.mat']);
xmin = all(1);
xmax = all(2); 
ymin = all(3);
ymax = all(4);

data=imread([loc,'sample.png']);
data= rgb2gray(data);
data=data(1:10:end,1:10:end);

% return
[x,y]=find(data>100);

[xall,yall]=find(data>-1);
zall= xall*0;
zall(find(data>100))=100;
ymod=y;
xmod=-x;

ymodall=yall;
xmodall=-xall;
%% find center
ymean=mean(ymodall);
xmean=mean(xmodall);
xscaleall=(xmodall-xmin)/(xmax-xmin);%(max(rr)/sqrt(2));
yscaleall=(ymodall-ymin)/(ymax-ymin);%(max(rr)/sqrt(2));
rrsel=sqrt((xmod-(xmin+xmax)/2).^2+(ymod-(ymin+ymax)/2).^2);
rr=sqrt((xmodall-(xmin+xmax)/2).^2+(ymodall-(ymin+ymax)/2).^2);%rrscale=rr./max(rrsel);
% rr=sqrt((xscaleall).^2+(yscaleall).^2);
%zall(find(rrscale>0.92))=0;
% % plot3(xall,yall,zall,'o')%rrscale)

%f = fit([xscaleall, yscaleall],zall,"linear");
f=0;
datatemp=[xscaleall,yscaleall,zall];
% % save datatemp datatemp

% % size(xscaleall)
% % size(yscaleall)
% % size(zall)
% return
% figure
% plot(f,[xscaleall yscaleall],zall)
% axis equal
% xlim([0 1])
% ylim([0 1]);
[noelkir,iiall]=checkmeshvae_func(xscaleall, yscaleall,zall,outerRadius,ssize);
return

% mean(mean(data))
% unique(data)
figure
contourf(reshape(ymodall,[size(data,1),size(data,1)]),reshape(xmodall,[size(data,1),size(data,1)]),data);

return


%%
% figure
% plot(ymod,xmod,'o');

xscale=(xmod-xmin)/(xmax-xmin);%(max(rr)/sqrt(2));
yscale=(ymod-ymin)/(ymax-ymin);%(max(rr)/sqrt(2));
% figure
% plot(yscale,xscale,'o')
% axis equal
% xlim([0 1])
% ylim([0 1]);


%% which element is the kirigami??
