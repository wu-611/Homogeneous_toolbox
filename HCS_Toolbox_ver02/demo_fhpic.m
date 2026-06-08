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

A= [0    1    0    0    0;
    0    0    0    1    0;
    0    0    0    0    1;
    0    0    0    0    0;
    0    0    0    0    0];

B= [0 0;
    0 0;
    0 0;
    1 0;
    0 1];

n=5;m=2;  

%%%%%%%%%%%%%%%%%%
%% FHPC Design
%%%%%%%%%%%%%%%%%%

[K0 K Ki G0 P mu1 mu2]=fhpic_design(A,B); % design  of FHPC

Ki=Ki*5;  %the integral gain can be multiplied by any positive definite matrix (m x m) to tune the convergence rate 

Gd1=eye(n)+mu1*G0; 
Gd2=eye(n)+mu2*G0;

%K0  - homogenization feedback gain
%K   - proportional gain
%Ki  - integral gain
%Gd1 - generator of dilation d1
%Gd2 - generator of dilation d2
%mu1 - negative homogeneity degree (close to 0)
%mu2 - positive homogeneity degree (close to Inf) 
%P   - shape matrix of the weighted Euclidean norm


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=10; 
h=0.005; % sampling period

x=[1;0;-1;0;0];
tl=[t];xl=[x];ul=[];

v=0; % for computation of integral term

noise=0; %magnitude of measurement noises

p=1; %contstant perturbation with an unknown bound

disp('Run numerical simulation...');

while t<Tmax
    xm=x+2*noise*(rand(n,1)-0.5);    
    
    %u=(K0+K)*xm;                           %linear P control (for comparison)
    
    %u=(K0+K)*xm; ui=Ki*xm;                 %linear PI control (for comparison)
   
    [u ui]=e_fhpic(xm,K0,K,Ki,G0,mu1,mu2,P);%FHPIC
    
    u=u+v; v=v+h*ui;  
    
    %u=e_fhpc(xm,K0,K,G0,P,mu1,mu2);        %FHPC (for comparison)
    
    x=x+h*A*x+h*B*(u+p);                    % simulation of the system 
    
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
title({'FHPIC,m=2'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-10 10]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
legend2 = legend(axes2,'show');
set(legend2,'Interpreter','latex');
