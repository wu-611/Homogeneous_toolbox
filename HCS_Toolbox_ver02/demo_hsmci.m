%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Example of disign of Homogenenous SMC with Integral Action (HSMCI)
%   
%  System:      dx/dt=A*x+B*(u+gamma(t,x)+p)  
%                
%        where  
%               x - system  state vector (n x 1)
%               u - control input (m x 1)
%               A - system matrix (n x n)
%               B - control matrix (m x m)
%               gamma(t,x) - unknown perturbation (vaninshing at x=0)
%               p - unknown constant (with unknown bound)
%  
% Control aim: 
%
%  Given matrix C (k x n) and gamma_max the goal is to design SMC such that 
%  
%          Cx(t)=0 for all t>=T,   where T is a finite time
%
%  for any p=const and |gamma(t,x)|<=gamma_max*sqrt(ncx)     
%  where ncx - homogeous norm of C*x
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
%% Model of system
%%%%%%%%%%%%%%%%%%%%%%%

A=[0 3 0 0 0;
   -3 0 0 0 0;
   0 0 1 -1 1;
   0 1 0 2 1;
   1 0 1 -1 3];  %  system matrix

B=[0 0;
   0 0;
   0 0;
   1 0;
   0 1]; % control matrix

C=[0 0 1 0 0;
   0 0 0 1 0;
   0 0 0 0 1]; % sliding surface  Cx=0

E=[0;0;1;0;0]; 

n=5;m=2;

%%%%%%%%%%%%%%%%%%
%% H2SMC design
%%%%%%%%%%%%%%%%%%
rho=1; %convergence rate tuning parameter (larger rho, faster the convergence) 
gamma_max=0.5; %magnitude of perturbation


[K0 K Ki Gd P]=hsmci_design(A,B,C,rho,gamma_max); %design the parameteres of HSMC 

%K0 - homogenization gain
%K  - proportional gain
%Ki - integral gain 
%Gd - generator of dilation
%P  - shape matrix of the weighted Euclidean norm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=5; 
h=0.001; % sampling period

x=[1;0;2;0;1];
tl=[t];xl=[x];ul=[];



  alpha=0; % no chattering attentuation
 %alpha=0.005;  % in the noise-free case
% alpha=0.1;  % for the noise magnitude 0.001;

p=1; %contstant perturbation with an unknown bound


noise=0;    % magnitude of measurement noises 

v=0; % for computation of integral term

disp('Run numerical simulation...');
[Ah,Bh]=ZOH(h,A,B);
[Ah_,Eh]=ZOH(h,A,E);

while t<Tmax
    xm=x+2*noise*(rand(5,1)-0.5);                                          %(possibly) noised state measurement


    [u ui]=e_hsmci(xm,C,K0,K,Ki,Gd,@(y)hnorm(y,Gd,P),alpha);              %HSMCI
    
    %u=(K0+K*C)*xm; ui=Ki*C*xm;                                            %linear PI control (for comparison)

    u=u+v; v=v+h*ui;                 

    %u=e_hsmc(xm,C,K0,K,Gd,@(y)hnorm(y,Gd,P),alpha);                      %HSMC (for comparison reasons)

    gamma_t=p+abs(x(5))*gamma_max*[sin(2*t);cos(10*t)];                    % matched perturb. 

    g_t=0.0*norm(C*x);                                                     % mismatched perturbation 
                                                                           % for the robustness test

    x=Ah*x+Bh*(u+gamma_t)+Eh*g_t;                              %simulation 

    t=t+h;
    tl=[tl t];
    xl=[xl x];
    ul=[ul u];
end;
ul=[ul u];

disp('Done!');

%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot simulation results
%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
axes1 = subplot(1,2,1);
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
title({'H2SMC,m=2'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-15 15]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
legend2 = legend(axes2,'show');
set(legend2, 'Interpreter','latex');



    