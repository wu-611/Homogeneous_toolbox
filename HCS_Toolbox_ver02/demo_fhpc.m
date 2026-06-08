%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Example of Fixed-time Homogenenous Proportional Control (FHPC)  design
%%   
%%  System:      dx/dt=A*x+B*u  
%%                
%%        where  
%%               x - system  state vector (n x 1)
%%               u - control input (m x 1)
%%               A - system matrix (n x n)
%%               B - control matrix (m x m)
%%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
%% Model of system
%%%%%%%%%%%%%%%%%%%%%%%
A= [28    26    15    93    28;
    68    69    59    49    98;
    90    13     7   65     4;
    91    12    82    89    33;
    75    19    72    54    97]/40;

B= [10 33;
    20 90;
    30 50;
    40 62;
    50 58]/40;

n=5;m=2;  

%%%%%%%%%%%%%%%%%%
%% FHPC Design
%%%%%%%%%%%%%%%%%%

[K0 K G0 P mu1 mu2 rho]=fhpc_design(A,B); % design  of FHPC

Gd1=eye(n)+mu1*G0;
Gd2=eye(n)+mu2*G0;

%K0 - homogenization feedback gain
%K  - control gain
%Gd1 - generator of dilation d1
%Gd2 - generator of dilation d2
%mu1 - negative homogeneity degree (close to 0)
%mu2 - positive homogeneity degree (close to Inf) 
%P  - shape matrix of the weighted Euclidean norm

%settling time estimate
T=1/(-mu1*rho)+1/(mu2*rho)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=8; 
h=0.01; % sampling period

x=[1;0;0;0;0];
tl=[t];xl=[x];ul=[];


noise=0; %magnitude of measurement noises

disp('Run numerical simulation...');

[Ah, Bh]=ZOH(h,A,B);

while t<Tmax
    xm=x+2*noise*(rand(n,1)-0.5);    
    
    %u=(K0+K)*xm;                                                          %linear control (for comparison)
    
   
    u=e_fhpc(xm,K0,K,G0,mu1,mu2,P,0.05);                                   %explicit discretization of FHPC

    %u=si_fhpc(h,xm,A,B,K0,K,G0,mu1,mu2,P);                                %semi-impicit discretization of FHPC


    x=Ah*x+Bh*u;                                                           %simulation of the system 
    
    t=t+h;
    tl=[tl t];
    xl=[xl x];
    ul=[ul u];
end;
ul=[ul u];
disp('Done!');

%%norm of the state at the time instant Tmax
disp(['||x(Tmax)||=',num2str(norm(x))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the simulation results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

axes1 = subplot(1,2,1);
%Plot the results
hold(axes1,'on');
plot1 = plot(tl,xl,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','$x_1$');
set(plot1(2),'DisplayName','$x_2$');
set(plot1(3),'DisplayName','$x_3$');
set(plot1(4),'DisplayName','$x_4$');
set(plot1(5),'DisplayName','$x_5$');
ylabel('$x$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'n=5'});
xlim(axes1,[0 Tmax]);
ylim(axes1,[-5 5]);
box(axes1,'on');
hold(axes1,'off');
set(axes1,'FontSize',30,'XGrid','on','YGrid','on');
legend1 = legend(axes1,'show');
set(legend1,'Interpreter','latex');


axes2 = subplot(1,2,2);
hold(axes2,'on');
plot2 = plot(tl,ul,'LineWidth',2);
set(plot2(1),'DisplayName','$u_1$');
set(plot2(2),'DisplayName','$u_2$');
ylabel('$u$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'FHPC,m=2'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-10 10]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
legend2 = legend(axes2,'show');
set(legend2,'Interpreter','latex');
