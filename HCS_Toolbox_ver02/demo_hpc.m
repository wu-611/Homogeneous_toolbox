%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Example of Homogenenous Proportional Controll (HPC) design
%%   
%%  System:      dx/dt=A*x+B*u,
%%                
%%        where  
%%               x - system  state vector (n x 1)
%%               u - control input (m x 1)
%%               A - system matrix (n x n)
%%               B - control matrix (m x m)
%%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model of system (harmonic oscillator)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A= [0 1; -1 0];                                                            % sysytem matrix 

B= [0;    1];                                                              % control input matrix

n=2; m=1;     % dimensions

%controlability test
W=[B];R=B; tn=1;
while (rank(W)<n) && (tn<=n)
 R=A*R;
 W=[W R];
 tn=tn+1;
end;
if tn>n disp('Error: the pair (A,B) is not controllable'); return; 
elseif norm(W)*norm(inv(W))>1e3 disp('Warning: the controllability matrix has a large condition number => consistent discrization is prooply conditioned'); 
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HPC design
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rho=1;                                                                     %convergence time 
                                                                           %tuning parameter 
                                                                           %larger rho => faster convergence 


mu=-0.5;                                                                   % the homogeneity degree  
                                                                           % mu<0  - finite-time stability, 
                                                                           % mu>0  - nearly fixed-time stability
                                                                           % mu=-1 - Sliding Mode Control (SMC)

[K0 K Gd P]=hpc_design(A,B,mu,rho);                                        % design  of HPC
                                                                           %K0 - homogenization feedback gain
                                                                           %K  - control gain
                                                                           %Gd - generator of dilation
                                                                           %P  - shape matrix of the norm 
                                                                           %        ||x||=sqrt(x'*P*x)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0; Tmax=3;                                                               %time inverval for simulation
h=0.01;                                                                   %sampling period

x=[1;0];
tl=[t];xl=[x];tul=[];ul=[];
if mu<0                                                                    %computation of the setting time
    T0=hnorm_newton(x,Gd,P)^(-mu)/(-mu*rho);
    disp(['Settling Time: T0=',num2str(T0)]);
end;


noise=0.00;                                                                %magnitude of measurement noises

disp('Run numerical simulation...');

[Ah, Bh]=ZOH(h,A,B);

while t<Tmax
    xm=x+2*noise*(rand(n,1)-0.5);                                          %modeling of noised measuremnts
    
    %u=(K0+K)*xm;                                                          %linear control (for comparison)
    


    u=e_hpc(xm,K0,K,Gd,mu,@(x)hnorm(x,Gd,P),0.01);                        % explicit discretization of  
                                                                           % the hom. control with
                                                                           % computation of hom. norm
                                                                           % by bisection method

    %u=si_hpc(h,xm,A,B,K0,K,Gd,mu,@(x)hnorm_newton(x,Gd,P),0.01);          % semi-implicit discretization
                                                                           % the hom. control with
                                                                           % computation of hom. norm
                                                                           % by Newton method

    %u=c_hpc(h,tn,xm,A,B,K0,K,Gd,mu,rho,@(x)hnorm_newton(x,Gd,P));         %consistent discretization
                                                                           %with computation of hom. norm  
                                                                           %by bisection method
   
    x=Ah*x+Bh*(u+0*sin(10*t));                                                           %simulation of the system 

                                                                           
    tul=[tul t t+h]; ul=[ul u u];                                          %store the control signal
    t=t+h;
    tl=[tl t]; xl=[xl x];                                                  %store the state vector
       
end;

disp('Done!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Evaluation of stabilization precision
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
disp(['|x(Tmax)|=',num2str(norm(x))])                                      % norm of the state at t= Tmax

disp(['||x([2.5,Tmax])||_2=',num2str(lp_norm(tl,xl,2.5,Tmax,2))])          %L_2 norm of solution on [2.5,Tmax]

disp(['||x([2.5,Tmax])||_Inf=',num2str(lp_norm(tl,xl,2.5,Tmax,Inf))])      %L_Inf norm of solution on [2.5,Tmax]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot of simulation results
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
xlim(axes1,[0 Tmax]);
ylim(axes1,[-2 2]);
box(axes1,'on');
hold(axes1,'off');
set(axes1,'FontSize',30,'XGrid','on','YGrid','on');
legend1 = legend(axes1,'show');
set(legend1,'Interpreter','latex');


axes2 = subplot(1,2,2);
hold(axes2,'on');
plot2 = plot(tul,ul,'k-','LineWidth',2);
ylabel('$u$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'HPC,m=1'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-2 2]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');