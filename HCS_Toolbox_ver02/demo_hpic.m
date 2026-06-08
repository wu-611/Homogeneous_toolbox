%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Example of Homogenenous Proportional Integral Control (HPIC) design
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

%%%%%%%%%%%%%%%%%%%
%% Model of system
%%%%%%%%%%%%%%%%%%%
A= [0 1; -1 0]; % sysytem matrix (harmonic oscillator)

B= [0;    1];  % control matrix

n=2; m=1;
 

%%%%%%%%%%%%%%%%%%
%% HPIC design
%%%%%%%%%%%%%%%%%%

mu=-0.5;   %homogeneity degree

[K0, K, Ki, Gd, P]=hpic_design(A,B,mu);

%K0 - homogenization feedback gain
%K  - proportional control gain
%Ki - integral control gain 
%Gd - generator of dilation
%P  - shape matrix of the weighted Euclidean norm


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=4; 
h=0.001; % sampling period

x=[1;0];
tl=[t];xl=[x];ul=[];


alpha=0.00;%chattering attenuation parameter

noise=0; %magnitude of measurement noises

disp('Run numerical simulation...');

p=1; %contstant perturbation with an unknown bound

v=0; % for computation of integral term
 
[Ah,Bh]=ZOH(h,A,B);
while t<Tmax
    xm=x+2*noise*(rand(n,1)-0.5);    
    
    %u=(K0+K)*xm;  ui=Ki*xm;                                               %linear PI control (for comparison)
    
    [u ui]=e_hpic(xm,K0,K,Ki,Gd,mu,@(x)hnorm(x,Gd,P),alpha);               %HPIC    
    
    u=u+v; v=v+h*ui;                     
    
    %u=e_hpic(xm,K0,K,Ki,Gd,mu,@(x)hnorm(x,Gd,P),alpha);                   %HPC (for comparison)


    x=Ah*x+Bh*(u+p);                                                        % simulation of the system
    t=t+h;
    tl=[tl t];
    xl=[xl x];
    ul=[ul u];
end;
ul=[ul u];

disp('Done!');

%%norm of the state at the time instant Tmax
disp(['||x(Tmax)||=',num2str(norm(x))])


%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot simulation results
%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
axes1 = subplot(1,2,1);
hold(axes1,'on');
plot1 = plot(tl,xl,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','$x_1$');
set(plot1(2),'DisplayName','$x_2$');
ylabel('$x$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'n=2'});
xlim(axes1,[0 Tmax]);
ylim(axes1,[-2 2]);
box(axes1,'on');
hold(axes1,'off');
set(axes1,'FontSize',30,'XGrid','on','YGrid','on');
legend1 = legend(axes1,'show');
set(legend1,'Interpreter','latex');


axes2 = subplot(1,2,2);
hold(axes2,'on');
plot2 = plot(tl,ul,'k-','LineWidth',2);
ylabel('$u$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'HPIC,m=1'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-2 2]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
