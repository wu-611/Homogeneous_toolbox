%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Example of the disign of a Homogenenous Sliding Mode Control (HSMC) for
%   
%  System:      dx/dt=A*x+B*(u+gamma(t))  
%                
%        where  
%               x - system  state vector (n x 1)
%               u - control input (m x 1)
%               A - system matrix (n x n)
%               B - control matrix (m x m)
%        gamma(t) - unknown bounded perturbation
%  
% Control aim: 
%
%    Given matrix C (k x n) the goal is to design a control law such that
%    independently of a bounded function gamma (with |gamma(t)|<gamma_max)
%          Cx(t)=0 for all t>=T,   where T is a finite time
%     
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

E=[0; 0; 1; 0; 0];  

n=5;m=2;
rho=1; %convergence rate tuning parameter (learger rho faster the convergence) 
gamma_max=0.5; %magnitude of perturbation


[K0 K Gd P]=hsmc_design(A,B,C,rho,gamma_max); %design the parameteres of HSMC 

%K0 - homogenization gain
%K  - control gain
%Gd - generator of dilation
%P  - shape matrix of the weighted Euclidean norm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=4; 
h=0.001; % sampling period

x=[1;0;2;0;1];
tl=[t];xl=[x];ul=[];

disp(['Settling Time=',num2str(hnorm(C*x,Gd,P)/rho)]);


 alpha=0; % no chattering attentuation
 %alpha=0.005;  % in the noise-free case
 %alpha=0.1;  % for the noise magnitude 0.001;


noise=0.000;    % magnitude of measurement noises 

disp('Run numerical simulation...');

[Ah, Bh]=ZOH(h,A,B);

[Ah_, Eh]=ZOH(h,A,E);


while t<Tmax
    xm=x+2*noise*(rand(5,1)-0.5);                                          %(possibly) noised state

    %u=(K0+K*C)*xm;                                                        %linear control (for comparison)
      
    u=e_hsmc(xm,C,K0,K,Gd,@(y) hnorm(y,Gd,P),alpha);                       %explicit discretization of HSMC
    
    %u=si_hsmc(h,xm,A,B,C,K0,K,Gd,P,alpha);                                %semi-implicit discret. of HSMC
    
    gamma_t=gamma_max*[sin(2*t);cos(10*t)];                                %matched perturbation 

    g_t=0.0*Eh*norm(C*x);                                                  %mismatched perturbation 

    x=Ah*x+Bh*(u+gamma_t)+g_t;                                             %simulation of the system

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
ylim(axes1,[-3 3]);
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
title({'HSMC,m=2'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-12 12]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');
legend2 = legend(axes2,'show');
set(legend2, 'Interpreter','latex');



    