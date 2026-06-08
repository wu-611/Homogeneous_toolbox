%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example of upgrading Linear Observer(LO) to Homogenenous Observer (HO) 
%%   
%% System:      dx/dt=A*x+f,    y=C*x
%%                
%%        where  
%%               x - system  state vector (n x 1)
%%               f - measured input (n x 1)
%%               y - measured output (k x 1)
%%               A - system matrix (n x n)
%%               C - output matrix (k x n)
%%      
%% The system is a linearized model of rotary inverted pendulum QUBE-Servo2 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pendulum model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Motor
% Resistance
Rm = 8.4;
% Current-torque (N-m/A)
kt = 0.042;
% Back-emf constant (V-s/rad)
km = 0.042;
%
%% Rotary Arm
% Mass (kg)
mr = 0.095;
% Total length (m)
r = 0.085;
% Moment of inertia about pivot (kg-m^2)
Jr = mr*r^2/3;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
br = 1e-3; % damping tuned heuristically to match QUBE-Sero 2 response
%
%% Pendulum Link
% Mass (kg)
mp = 0.024;
% Total length (m)
Lp = 0.129;
% Pendulum center of mass (m)
l = Lp/2;
% Moment of inertia about pivot (kg-m^2)
Jp = mp*Lp^2/3;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
bp = 5e-5; % damping tuned heuristically to match QUBE-Sero 2 response
% Gravity Constant
g = 9.81;

% Total Inertia
Jt = Jr*Jp - mp^2*r^2*l^2;
% 

%% Linearized model of the rotary inverted pendulum (in the upper position):
%%
%%           dx/dt=Ax+Bu,      x=(x1,x2,x3,x4)'
%%
%% where     x1  - angle of the pentulum arm
%%           x2  - angle of the rotary arm
%%           x3  - angular velocity of the pendulum arm
%%           x4  - angular velocity of the rotary arm

A = [0 0 1 0;
     0 0 0 1;
     0 mp^2*l^2*r*g/Jt  -br*Jp/Jt   -mp*l*r*bp/Jt 
     0  mp*g*l*Jr/Jt    -mp*l*r*br/Jt   -Jr*bp/Jt];
%
B = [0; 0; Jp/Jt; mp*l*r/Jt];

% adding a model of actuator dynamics
A(3,3) = A(3,3) - km*km/Rm*B(3);
A(4,3) = A(4,3) - km*km/Rm*B(4);
B = km * B / Rm;

C=[1 0 0 0;
    0 1 0 0]
n=4;m=1;k=2;

%T=block_obs(A,C)
%At=inv(T)*A*T
%Ct=C*T; 
%
%Lt=[6^2 152.0057;
%    0 +264+6^2;
%    0  0.5005;
%   12.1117 -11.2415];
%
%L_lin=T*Lt;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HO/FHO design by upgrading a linear observer 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the gain of the linear (Luenberger) observer (should be provided by manufacturer)

L_lin=[ 0    0.5005;
   12.1117  -11.2415;
  -42.0617 -152.5127;
  -10.5401 -296.2791];


[L0 G0 nu1 nu2]=lo2ho(A,C,L_lin); % upgrade linear control to HPC

%selection of the homogeneity degree

L=L_lin-L0;   

%L0 - homogenization gain
%L  - observer gain gain
%G0 - matrix that defines the generator Gd=eye(n)+nu*G0 of dilation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HPC/FHPC design by upgrading a linear controller 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Klin=[2 -35 1.5 -3];

 % the linear feedback gain (provided by manufacturer)


[K0 G0_con P mu1 mu2]=lpc2hpc(A,B,Klin); % upgrade linear control to HPC

K_con=Klin-K0;   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=4; 
h=0.02; % sampling period

x=[2;2;1;2]; z=[0;0;0;0];
tl=[t];xl=[x];zl=[z];ul=[];

%alpha and beta are tuning parameters of the upgrade
alpha=0.4; beta=1;   %upgrade close to zero
%alpha=1;beta=Inf;    %upgrade close to Inf)                    
%alpha=0.000;beta=Inf;   %global upgrade                    

noise=0.00; %magnitude of measurement noises (may be changed for comparison)

disp('Run numerical simulation...');
[Ah,Bh]=ZOH(h,A,B); [Ac,Ic]=ZOH(h,A+L_lin*C,eye(n));
while t<Tmax
    
    u=Klin*z; % generated input to the plant (observer-based feedback)
    
    %u=e_fhpc(z,K0,K_con,G0_con,P,mu1,mu2,0.05,100);  %FHPC
    
    y=C*x+2*noise*(rand(k,1)-0.5); %measured output
    
    
    x=Ah*x+Bh*u; %simulation of the plant
    
    
    %z=z+h*(A*z+B*u+L_lin*(C*z-y)); %original observer expl (for comparison)
    %z=Ac*z+Ic*(B*u-L_lin*y);       %original observer ZOH (for comparison)
    
   
    z=e_ho(h,z,y,A,C,B*u,L0,L,eye(n)+nu1*G0,nu1,alpha,beta); %HO
    %z=si_ho(h,z,y,A,C,B*u,L0,L,eye(n)+nu1*G0,nu1,alpha,beta); %HO

    %z=e_fho(h,z,y,A,C,B*u,L0,L,G0,nu1,nu2,alpha,beta);  %FHO
     
    t=t+h;
    tl=[tl t];
    xl=[xl x];
    zl=[zl z];
    ul=[ul u];
end;
ul=[ul u];


disp('Done!');

%%%norm of the error at the time instant Tmax
disp(['||x(Tmax)||=',num2str(norm(x))])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot simulation results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

axes1 = subplot(1,3,1);
hold(axes1,'on');
plot1 = plot(tl,xl,'LineWidth',2,'Parent',axes1);
set(plot1(1),'DisplayName','$x_1$');
set(plot1(2),'DisplayName','$x_2$');
set(plot1(3),'DisplayName','$x_3$');
set(plot1(4),'DisplayName','$x_4$');
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


axes2 = subplot(1,3,2);
hold(axes2,'on');
plot2 = plot(tl,xl-zl,'LineWidth',2);
set(plot2(1),'DisplayName','$x_1-z_1$');
set(plot2(2),'DisplayName','$x_2-z_2$');
set(plot2(3),'DisplayName','$x_3-z_3$');
set(plot2(4),'DisplayName','$x_4-z_4$');
ylabel('$x-z$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'FHO Error,k=2'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-10 10]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
legend2 = legend(axes2,'show');
set(legend2,'Interpreter','latex');


axes3 = subplot(1,3,3);
hold(axes3,'on');
plot3 = plot(tl,ul,'LineWidth',2);
ylabel('$u$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'FHPC,m=1'});
xlim(axes3,[0 Tmax]);
ylim(axes3,[-10 10]);
box(axes3,'on');
hold(axes3,'off');
set(axes3,'FontSize',30,'XGrid','on','YGrid','on');
