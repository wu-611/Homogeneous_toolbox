%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Example of Fixed-Time Homogenenous Observer (HO) design
%%   
%%  System:      dx/dt=A*x+B*u, y=C*x
%%                
%%        where  
%%               x - system  state vector (n x 1)
%%               u - control input (m x 1)
%%               y - measured output (k x 1)
%%               A - system matrix (n x n)
%%               B - control matrix (m x m)
%%               C - output matrix (k x n)
%%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%
%% Model of system
%%%%%%%%%%%%%%%%%%%

A= [0 1; -1 0]; % sysytem matrix (harmonic oscillator)

C= [1  0];  % output matrix

n=2; m=1; k=1;

%%%%%%%%%%%%%%%%%
%% HO design
%%%%%%%%%%%


[L0, L, G0, nu1, nu2]=fho_design(A,C); % design  of HPC

%L0 - homogenization gain
%L  - observer gain
%G0 - defines generators of dilations Gd1=eye(n)+nu1*G0; Gd2=eye(n)+nu2*G0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=4; 
h=0.001; % sampling period

x=[10;0]; z=[0;0];
tl=[t];xl=[x];zl=[z];

alpha=0.0;%tuning parameter
beta=Inf;%tuning parameter

noise=0; %magnitude of measurement noises

disp('Run numerical simulation...');

while t<Tmax
    
    
    x=x+h*A*x; % explicit Euler method (system dynamics)
 
    y=C*x+2*noise*(rand(k,1)-0.5); %measurement of the system

    
    z=e_fho(h,z,y,A,C,0,L0,L,G0,nu1,nu2,alpha,beta);%explicit  FHO
  
      
    %z=si_fho(h,z,y,A,C,0,L0,L,G0,nu1,nu2,alpha,beta);%semi-implicit FHO
  
    %z=zoh_fho(h,z,y,A,C,0,L0,L,G0,nu1,nu2,alpha,beta);%semi-implicit FHO


    t=t+h;
    tl=[tl t];
    xl=[xl x];
    zl=[zl z];
end;

disp('Done!');

%%norm of the error at the time instant Tmax
disp(['||z(Tmax)-x(Tmax)||=',num2str(norm(x-z))])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot simulation results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

axes1 = subplot(1,2,1);
hold(axes1,'on');
plot1 = plot(tl,xl,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','$x_1$');
set(plot1(2),'DisplayName','$x_2$');
ylabel('$x$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'n=2'});
xlim(axes1,[0 4]);
ylim(axes1,[-10 10]);
box(axes1,'on');
hold(axes1,'off');
set(axes1,'FontSize',30,'XGrid','on','YGrid','on');
legend1 = legend(axes1,'show');
set(legend1,'Interpreter','latex');


axes2 = subplot(1,2,2);
hold(axes2,'on');
plot2 = plot(tl,xl-zl,'LineWidth',2);
set(plot2(1),'DisplayName','$x_1-z_1$');
set(plot2(2),'DisplayName','$x_2-z_2$');
ylabel('$x-z$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'FHO,k=1'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-10 10]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
legend2 = legend(axes2,'show');
set(legend2,'Interpreter','latex');