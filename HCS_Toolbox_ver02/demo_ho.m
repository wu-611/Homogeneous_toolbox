%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Example of Homogenenous Observer (HO) design
%%   
%%  System:      dx/dt=A*x+f, y=C*x
%%                
%%        where  
%%               x - system  state vector (n x 1)
%%               f - measured input (n x 1)
%%               y - measured output (k x 1)
%%               A - system matrix (n x n)
%%               C - output matrix (k x n)
%%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%
%% Model of system
%%%%%%%%%%%%%%%%%%%

A= [0 1; -1 0]; % sysytem matrix (harmonic oscillator)

C= [1  0 ];  % output matrix

n=2;  k=1;

%%%%%%%%%%%%%%%%%
%% HO design
%%%%%%%%%%%

rho=1; %convergence rate tuning parameter (larger rho, faster convergence) 

nu=-0.5; % the homogeneity degree  
         %     nu<0 - finite-time stability, 
         %     nu>0 - nearly fixed-time stability

[L0 L Gd, P]=ho_design(A,C,nu,rho); % design  of HPC

A0=A+L0*C;
%L0 - homogenization gain
%L  - observer gain
%Gd - generators of dilation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta=0.01;
t=0; Tmax=6; % time interval
h=0.001; % sampling period
x=[1;0]; z=[0;0];
tl=[t];xl=[x];zl=[z]; hn_l=[hnorm(x-z,Gd,P)];
xi=max(delta,abs(C*x-C*z));
n_l=[sqrt((expm(-log(xi)*Gd)*(x-z))'*P*expm(-log(xi)*Gd)*(x-z))];
c_l=[abs(C*x-C*z)];
cA0_l=[abs(C*A0*x-C*A0*z)];

alpha=0.000;%tuning parameter
beta=Inf;%tuning parameter


noise=0.000; %magnitude of measurement noises

disp('Run numerical simulation...');
G0=(Gd-eye(n))/nu;
while t<Tmax
    
    
    x=x+h*A*x; % explicit Euler method (system dynamics)
 
    y=C*x+2*noise*(rand(k,1)-0.5); %measurement of the system
    %z=e_ho(h,z,y,A,C,0,L0,L,Gd,nu,alpha,beta); %explicit discretization of HO
    z=si_ho(h,z,y,A,C,0,L0,L,Gd,nu,alpha,beta);  %semi-implicit discretization of HO
    
    %z=e_ho(h,z,y,A,C,0,L0,L,Gd,nu,1,1);;  %linear observer  (semi

    t=t+h;
    tl=[tl t];
    xl=[xl x];
    zl=[zl z];
    hn_l=[hn_l hnorm(x-z,eye(n)+nu*G0,P)];
    xi=max(delta,abs(C*x-C*z));
    n_l=[n_l, sqrt((expm(log(xi)*Gd)*(x-z))'*P*expm(log(xi)*Gd)*(x-z))];
    c_l=[c_l, abs(C*x-C*z)];
    cA0_l=[cA0_l, abs(C*A0*x-C*A0*z)];
end;

disp('Done!');

%%norm of the error at the time instant Tmax
disp(['||z(Tmax)-x(Tmax)||=',num2str(norm(x-z))])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gn=-3/4;
figure 
hold on
%plot(tl,hn_l);
%plot(tl,0.45*cA0_l.^(1/(1+nu))+c_l);
plot(tl,c_l);
plot(tl,n_l);
return
%% Plot simulation results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

axes1 = subplot(1,3,1);
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


axes2 = subplot(1,3,2);
hold(axes2,'on');
plot2 = plot(tl,xl-zl,'LineWidth',2);
set(plot2(1),'DisplayName','$x_1-z_1$');
set(plot2(2),'DisplayName','$x_2-z_2$');
ylabel('$x-z$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'HO,k=1'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-2 2]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
legend2 = legend(axes2,'show');
set(legend2,'Interpreter','latex');


