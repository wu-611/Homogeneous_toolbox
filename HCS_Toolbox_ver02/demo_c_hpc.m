%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Example of Constant-time Homogenenous Proportional Control (C-HPC) 
%%   
%%  System:      dx/dt=A*x+B*u,  t>0,  x(0)=x0
%%                
%%        where  
%%               x - system  state vector (n x 1)
%%               u - control input (m x 1)
%%               A - system matrix (n x n)
%%               B - control matrix (m x m)
%%      
%% The control aim: stabilize x at zero exactly at the given time T>0
%%
%% The homogeneous constant time stabilizer:
%%
%%    u=K0*x+KT*d(-log(NT(x)))*x,    NT(x)=min(1,hnorm(x/sqrt(x0'*PT*x0)))
%%
%%  where KT - (m x n), PT  - (n x n) are matrices dependent of T and 
%%        d - is a linear dilation in R^n  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model of system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A= [0 1; -1 0]; % (n x n) sysytem matrix (harmonic oscillator)

B= [0;    1];  % (n x m) control matrix

n=2;  m=1;  %dimensions

%controlability test
W=[B];R=B; tn=1;
while (rank(W)<n) && (tn<=n)
 R=A*B;
 W=[W R];
 tn=tn+1;
end;
if tn>n disp('the pair (A,B) is not controllable'); return; end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Contant-time HPC design 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[K0 K Gd P]=hpc_design(A,B,-1,1);                          % design  of HPC

%K0 - homogenization feedback gain
%K  - control gain
%Gd - generator of dilation
%P  - shape matrix of the weighted Euclidean norm

inv(P)

T=1.5;                                           %the desired settling time
K=K*expm(-log(T)*Gd);                                  %scaling of the gain
P=expm(-log(T)*Gd)'*P*expm(-log(T)*Gd);        %scaling of the shape matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


t=0;                                                           %intial time

h=0.025;                                                    %sampling period

x=[0.5;0];  %variation of the intia state does not change the settling time

denom=sqrt(x'*P*x);   %initial-state-depenendent factor for control scaling

tl=[t];xl=[x];tul=[];ul=[];





noise=0.0000; %magnitude of measurement noises

disp('Run numerical simulation...');

[Ah, Bh]=ZOH(h,A,B);

Tmax=2; 

while t<Tmax
    xm=x+2*noise*(rand(n,1)-0.5); % model of measuremnts
        
    
    % contant-time homogeneous stabilizer
    if x'*P*x<= denom^2
    u=denom*c_hpc(h,tn,xm/denom,A,B,K0,K,Gd,-1,1,@(x)hnorm(x,Gd,P));       %consistent 
    %u=denom*e_hpc(xm/denom,K0,K,Gd,-1,@(x)hnorm(x,Gd,P));                  %explicit
    else
        u=(K0+K)*x;
    end;   

   % if t<=T
   %    Kt=(K0+K*exp(-Gd*log(T-t)));
   %    u=Kt*xm;                           % PT control by explicit Euler method (for comparison)
   %    u=Kt*inv(eye(2)-h*(A+B*Kt))*xm;    % PT control by implicit Euler method (for comparison)
   % else u=(K0+K)*x;
   % end;

    x=Ah*x+Bh*(u+sin(2*t)); %simulation of the system 

    %%store the results to the lists
    tul=[tul t t+h];
    t=t+h;
    tl=[tl t];
    xl=[xl x];
    ul=[ul u u];
end;

%ul=[ul u];

disp('Done!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Evaluation of stabilization precision

%%norm of the state at the time instant Tmax
disp(['|x(Tmax)|=',num2str(norm(x))])

%norm of the state at the time instant T0+n*h
%disp(['|x(T0+n*h)|=',num2str(norm(xl(:,round(T0/h)+n)))])

%%L2 norm of solution on [3,Tmax]
disp(['||x([3,Tmax])||_2=',num2str(lp_norm(tl,xl,T,Tmax,2))])

%%L_Inf norm of solution on [3,Tmax]
disp(['||x([3,Tmax])||_Inf=',num2str(lp_norm(tl,xl,T,Tmax,Inf))])

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
xlim(axes1,[0 Tmax]);
ylim(axes1,[-2 2]);
box(axes1,'on');
hold(axes1,'off');
set(axes1,'FontSize',30,'XGrid','on','YGrid','on');
legend1 = legend(axes1,'show');
set(legend1,'Interpreter','latex');


axes2 = subplot(1,2,2);
hold(axes2,'on');
plot2 = plot(tul,ul,'LineWidth',2,'Color',[0 0 0]);
ylabel('$u$','Interpreter','latex');
xlabel('$t$','Interpreter','latex');
title({'C-HPC,m=1'});
xlim(axes2,[0 Tmax]);
ylim(axes2,[-5 5]);
box(axes2,'on');
hold(axes2,'off');
set(axes2,'FontSize',30,'XGrid','on','YGrid','on');