%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Plot of  homogeneous curve 
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gd=[2 -1; 1 1]; % a dilation in R^2

%plotting homgoeneous curves

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');

sl=[-5:0.01:2];

N=15; %number of curves to plot
for i=0:N

x=[cos(2*pi*i/N);sin(2*pi*i/N)];          % vector x in R^2

zl=hcurve(x,Gd,sl);
plot(x(1),x(2),'*');
plot(zl(1,:),zl(2,:),'LineWidth',2);
end;

xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
 xlim(axes1,[-8 8]);
 ylim(axes1,[-4 4]);
box(axes1,'on');
set(axes1,'FontSize',30,'XGrid','on','XTick',[-5 0 5],'YGrid','on','YTick',...
    [-4 -2 0 2 4]);







