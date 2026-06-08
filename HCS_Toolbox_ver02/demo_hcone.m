%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Plot of  homogeneous cone
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gd=[1 0; 0 2]; % a dilation in R^2

%plotting homgoeneous curves

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');

P=[3 1; 1 2];

C=hcone([1 0;1 -1; -1 2],Gd,P,5000);                                       %generation of points
                                                                           %on the unit sphere
                                                                           %belonging to linear
                                                                           %homogeneous cone

sl=[-5:0.01:1];                                                            % a grid for generation 
                                                                           % of homogeneous curves 

for i=1:size(C,2)

zl=hcurve(C(:,i),Gd,sl);                                                   % generation of homogeneous curve
                                                                           %belonging to linear homogeneous cone
                                                                          
plot(zl(1,:),zl(2,:),'LineWidth',2,'Color',[1 0 0]);
end;

xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
 xlim(axes1,[-1 1]);
 ylim(axes1,[-1 1]);
box(axes1,'on');
set(axes1,'FontSize',30,'XGrid','on','XTick',[-1 -0.5 0 0.5 1],'YGrid','on','YTick',...
    [-1 -0.5 0 0.5 1]);








