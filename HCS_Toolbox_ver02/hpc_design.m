function [K0, K, Gd, P]=hpc_design(A, B, mu, rho, gamma, R)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [K0, K, Gd, P]=hpc_design(A, B, mu)                     
%% computes parameters of Homogeneous Proportional Controller (HPC)     
%%                                                                      
%%    u=K0*x+nx^(1+mu)*K*d(-log(nx))*x         (nx - homogeneous norm of x)         
%%                                                                      
%% which stabilizes the linear MIMO system                             
%%                                                                      
%%     dx/dt = A*x + B*u,   x(0) = x0                                   
%%                                                                      
%% (a) in a finite time if mu<0, namely,                                
%%                                                                      
%%     x(t) = 0 for all t>=nx0^(-mu)/(-mu)                  
%%                                                                      
%% (b) in a nearly fixed time if mu>0, namely,                                 
%%                                                                      
%%     nx(t)<=r for t>= 1/(mu*r^mu) and any r>0       (independently of x0)        
%%                                                                        
%% where nx0 - homogeneous norm of x0 and nx(t) - homogeneous norm of x(t)  
%%                                                                      
%% The input parameters:                                             
%%     A  - system matrix (n x n)                                       
%%     B  - control matrix (n x m)               
%%     mu - homogeneity degree (mu=-1 corresponds to sliding mode control)       
%%                                                                      
%% The output parameters:                                                
%%     K0 - feedback gain (m x n) of the linear term of HPC          
%%     K  - feedback gain (m x n) of the nonlinear term of HPC       
%%     Gd - anti-Hurwitz matrix (n x n) being a generator of the dilation 
%%                          d(s)=expm(s*Gd),   s - a real number                   
%%     P  - positive definite matrix which defines the weighted Euclidean 
%%         norm sqrt(x'*P*x) that induces the canonical homogeneous norm nx  
%%                                                                      
%% Hints:                                                                
%%  - to compute homogeneous norm nx use the function hnorm(x,Gd,P)
%%  - to implement HPC use the function hpc
%%  - to design a Homogenenous Proportional-Intergral Control (HPIC):
%%      u=uh+v,  dv/dt=ui                   <- u=proportional+integral  
%%     uh=K0*x+nx^(1+mu)*K*d(-log(nx))*x    <- HPC compontent 
%%     ui=nx^(1+2*mu)*Ki*d(-log(nx))*x/(x'*d'(-log(nx))*P*Gd*d(-log(nx))*x)
%%                         (^^^integrant^^^^)
%%   please use                                                                
%%                             Ki=-Q*B'*P                                                   
%%   where Q is an arbitrary positive definite matrix (m x m) 
%%                                                                      
%%                                                                      
%%                                                                      
%% The function [K0, K, Gd, P]=hpc_design(A, B, mu, rho)                
%% uses aditional input parameter rho for tuning  of convergence time:  
%%                                                                      
%% (a)  a finite time convergence if mu<0, namely,                      
%%                                                                      
%%          x(t) = 0 for all t>=h_norm(x0)^(-mu)/(-rho*mu)              
%%                                                                      
%% (b) a nearly fixed time convergence if mu>0, namely,                 
%%                                                                      
%%      h_nom(x(t))<=r for all t>= 1/(rho*mu*r^mu)  and for any r>0     
%%                                                                      
%%                                                                      
%%                                                                      
%% The function [K0, K, Gd, P]=hpc_design(A, B, mu, rho, gamma)         
%% uses additional parameter gamma to deal with matched perturbations:  
%%                                                                      
%%   p(t,x)=Bg(t,x) :  norm(g(t,x))<=gamma*h_norm(x)^(1+mu)             
%%                                                                      
%%                                                                      
%%                                                                      
%% The function [K0, K, Gd, P]=hpc_design(A, B, mu, rho, gamma, R)      
%% uses additional parameter R to deal with mismatched perturbations:   
%%                                                                      
%%    p'(t,x)*D'(x)*R*D(x)*p(t,x)<=gamma^2*h_norm(x)^(2*mu)             
%%                                                                      
%% where D(x)=d(-log(h_norm(x))) and d(s) is dilation (defined above).   
%%                                                                      
%% The function [K0, K, Gd, P]=hpc_design(A, B, mu, rho, gamma)         
%% uses additional parameter gamma to deal with matched perturbations:  
%%                                                                      
%%   p(t,x)=Bg(t,x) :  norm(g(t,x))<=gamma*h_norm(x)^(1+mu)             
%%                                                                      
%%                                                                      
%%                                                                      
%% The function [K0, K, Gd, P]=hpc_design(A, B, mu, rho, gamma, R, C)      
%% uses additional parameter R to deal with mismatched perturbations:   
%%                                                                      
%%    p'(t,x)*D'(x)*R*D(x)*p(t,x)<=gamma^2*h_norm(x)^(2*mu)             
%%                                                                      
%% where D(x)=d(-log(h_norm(x))) and d(s) is dilation (defined above).   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K0=0; K=0; Gd=0; P=0;
tol=0.00001; %tolerance

%1) Check the parameters

sA=size(A); sB=size(B);

if sA(1)~=sA(2) disp('Error: the system matrix must be square'); return;
end;
if sB(1)~=sA(2) disp('Error:  dimensions of system  and control matrices must agree');return;
end;
n=sB(1); m=sB(2);

%controllability check
U=[B];tA=A;
k=0;
if rank(U)==n k=1;
end;i=1;
while (k==0) && (i<=n-1)
    i=i+1;
    U=[U tA*B];
    if rank(U,tol)==n k=i; end;
    tA=A*tA;
end;
if k==0 disp('Error: The system is not controllable'); return;
end;

if nargin==2 mu=-1; end;

if mu>1/k disp('Error: The requested homogeneity degree is invalid.');
          disp('Please chose the homogeneity degree more close to 0 (see the documentation)');
          return;
end; 

if mu<-1  disp('Error: The requested homogeneity degree is invalid.');
          disp('Please select a homogeneity degree more close to 0 (see the documentation)');
          return;
end;

if nargin<=3 rho=1; end;
if  rho<=0 disp('Error: The convergence time tuning parameter rho must be positive.'); return; end;

if nargin<=4 gamma=0; end;

if gamma<0 disp('Error: The pertubation bound gamma must be non-negative.'); return; end;

if nargin<=5 R=B*B'; end;

if norm(R-R')>tol disp('Error: The shape pertubation matrix R must be symmetric.'); return; end;

if min(real(eig(R)))<-tol disp('Error: The shape pertubation matrix R must be non-negative definite.'); return; end;
    
if det(U*U')<tol  disp('Warning: The system is weakly controllable. Parameters may be badly tuned due to computations issues.');
end;    


R=gamma*R;

if rank(B)==n 
   K0=-B'*inv(B*B')*A;
   K=-B'*inv(B*B'); Gd=eye(n);
   P=eye(n);
   return;
end;   
    

%%%%%%%%%%%%%%%%%%%%%
% Design of control parameters using block decomposition
%%%%%%%%%%%%%%%%%%
[T nt]=block_con(A,B);
if T==0 K0=0; K=0;P=0; Gd=0; 
    disp('Error: A block decomposition is impossible.'); 
      return; end;
try
Anew=T*A*inv(T);
R=T*R*T';
k=size(nt,1); 
%creation of an array of indecies
s=1;
n_ind=[s];
for i=1:k-1
s=s+nt(i);
n_ind=[n_ind; s];
end;

%2) Compute tGd
vG0=[];
for i=1:k
 vG0=[vG0 mu*(k-i)*ones(1,nt(i))];   
end;
Gd=eye(n)-diag(vG0);
Gd_tmp=Gd;
Gd=(rho+0.5*gamma)*Gd;
%Gd=rho*(Gd-0.5*eye(n)); %to be deleted 

%3) Compute K0
Bnew=T*B;
B0=Bnew(n_ind(k):n,1:m);
K0=-B0'*inv(B0*B0')*Anew(n_ind(k):n,1:n);
A0=Anew+Bnew*K0;
K0=K0*T;

%3) Compute P and K 
Xii=eye(nt(1));X=[Xii];
Gii=Gd(n_ind(1):n_ind(1)+nt(1)-1,n_ind(1):n_ind(1)+nt(1)-1);
Zii=Xii*Gii'+Gii*Xii;
Rii=R(n_ind(1):n_ind(1)+nt(1)-1,n_ind(1):n_ind(1)+nt(1)-1);
Z=[Zii]; 
for i=1:k-1
Ai_i1=A0(n_ind(i):n_ind(i)+nt(i)-1,n_ind(i+1):n_ind(i+1)+nt(i+1)-1);
Xi_i1=-0.5*(Zii+Rii)*inv(Ai_i1*Ai_i1')*Ai_i1;
Xl=[Xi_i1];
j=i-1;
while j>0
   Aj_j1=A0(n_ind(j):n_ind(j)+nt(j)-1,n_ind(j+1):n_ind(j+1)+nt(j+1)-1);
   Gjj=Gd(n_ind(j):n_ind(j)+nt(j)-1,n_ind(j):n_ind(j)+nt(j)-1);   
   Xj1_i=X(n_ind(j+1):n_ind(j+1)+nt(j+1)-1,n_ind(i):n_ind(i)+nt(i)-1); 
   Xj_i=X(n_ind(j):n_ind(j)+nt(j)-1,n_ind(i):n_ind(i)+nt(i)-1);  
   Rj_i=R(n_ind(j):n_ind(j)+nt(j)-1,n_ind(i):n_ind(i)+nt(i)-1);
 Xl=[-(Aj_j1*Xj1_i+Gjj*Xj_i+Xj_i*Gii'+Rj_i)*inv(Ai_i1*Ai_i1')*Ai_i1; Xl];
j=j-1;
end;
%compute new Gii
G_=Gd(1:n_ind(i+1)-1,1:n_ind(i+1)-1);
Gii=Gd(n_ind(i+1):n_ind(i+1)+nt(i+1)-1,n_ind(i+1):n_ind(i+1)+nt(i+1)-1);
Rii=R(n_ind(i+1):n_ind(i+1)+nt(i+1)-1,n_ind(i+1):n_ind(i+1)+nt(i+1)-1);
Zl=Xl*Gii'+G_*Xl;
sg=min(real(eig(Gii'+Gii)));
lambda1=max(real(eig(Xl'*inv(X)*Xl)))+1;
lambda2=max(real(eig(Zl'*inv(Z)*Zl)))/sg+1;
lambda=max(lambda1,lambda2);
%compute Xii, X and Z for the next iteration
Xii=lambda*eye(size(Xl,2)); 
X=[X Xl; Xl' Xii];
Zii=Xii*Gii'+Gii*Xii;
Z=[Z Zl;
    Zl' Zii];
end;
tmpM=(A0+Gd)*X+X*(A0+Gd)'+R;
B0Y=[-tmpM(n_ind(k):n_ind(k)+nt(k)-1,1:n_ind(k)-1) -0.5*tmpM(n_ind(k):n_ind(k)+nt(k)-1,n_ind(k):n_ind(k)+nt(k)-1)];
Y=B0'*inv(B0*B0')*B0Y;
K=Y/X;
Gd=inv(T)*Gd_tmp*T;
X=inv(T)*X*inv(T)';
P=inv(X);
K=K*T;
catch
 K0=0; K=0;P=0; Gd=0; disp('Error: The control gains cannot be found with a block decomposition method');
end;

