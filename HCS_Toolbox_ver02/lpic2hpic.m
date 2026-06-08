function [K0, G0, P, Ki_new, mu_min, mu_max]=lpic2hpic(A,B,K,Ki)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [K0,G0,P,Ki_new,mu_min,mu_max]=lpc2hpc(A,B,K,Ki) upgrades
%% the linear proportional-integral controller (LPIC)
%% 
%% u=K*x+v,  dv/dt=Ki*x      (x - system state)   
%%
%% to the Homogeneous Proportional-Integral Controller (HPIC): 
%%                                                             
%%        u = uh + v,  dv/dt=ui   
%%
%%  uh=K0*x + nx^(1+mu)*(K-K0)*px            (nx - homogeneous norm of x)
%%  
%%  ui=nx^(1+2*mu)*Ki_new*p(x)/(px'*P*Gd*px)      
%%
%%  where  px=d(-log(nx))*x - homogenenous projection                                                                    
%% 
%%
%% The input parameters:                                             
%%      A  - system matrix (n x n)                                       
%%      B  - control matrix (n x m)       
%%      K  - proportional gain matrix (m x n) of LPIC s.t. A+B*K - Hurwitz  
%%      Ki - integral gain matrix (m x n) of LPIC 
%%
%% The output parameters:                                             
%%      K0 - feedback gain (m x n) of a linear part of HPC controller                                          
%%      G0 - matrix (n x n) that  defines the generator Gd=eye(n)+mu*G0  
%%           of the dilation d(s)=expm(s*Gd)                   
%%       P - positive definite matrix which defines the canonical        
%%           homogeneous norm nx induced by the norm sqrt(x'*P*x)
%%  Ki_new - redesigned gain of integral component of HPIC
%%  mu_min - minimal admissible degree of HPIC  (mu_min<=mu)  
%%  mu_max - maximal admissible degree of HPIC  (mu<=mu_max)
%%
%%
%% WARNING: the upgrade neends the full column rank of the control matrix B 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K0=0; G0=0; P=0; Ki_new=0; mu_min=0; mu_max=0;
tol=0.00001; %tolerance

%1) Check the parameters

sA=size(A); sB=size(B); sK=size(K);

if sA(1)~=sA(2) disp('Error: the system matrix must be square'); return;
end;
if sB(1)~=sA(2) disp('Error:  dimensions of system  and control matrices must agree');return;
end;
if sK(1)~=sB(2) disp('Error:  dimensions of control and gain matrices must agree');return;
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
    if rank(U)==n k=i; end;
    tA=A*tA;
end;
if k==0 disp('Error: The system is not controllable'); return;
end;
       
if rank(B)<m disp('Error: The control matrix must have full column rank.'); return; end;

rho=-max(real(eig([A+B*K B; Ki zeros(m,m)])));

if rho<tol disp('Error: The linear control system does not have a sufficient stability margin.'); return; end;
rho=min(rho,1);

if rank(B)==n 
   K0=-B'*inv(B*B')*A;
   Gd0=zeros(n,n);P=eyey(n);
   Ki_new=Ki;
   mu_min=-0.5; mu_max=1;
   return;
end;   
    

%%%%%%%%%%%%%%%%%%%%%
% Design of Gd using block decomposition
%%%%%%%%%%%%%%%%%%

[T nt]=block_con(A,B);
if T==0 K0=0; K=0;P=0; Gd=0; 
    disp('Error: A block decomposition is impossible.'); 
      return; end;
  
Anew=T*A*inv(T);
k=size(nt,1); 
%creation of an array of indecies
s=1;
n_ind=[s];
for i=1:k-1
s=s+nt(i);
n_ind=[n_ind; s];
end;

Bnew=T*B;
B0=Bnew(n_ind(k):n,1:m);
K0=-B0'*inv(B0*B0')*Anew(n_ind(k):n,1:n);
A0=Anew+Bnew*K0;
K0=K0*T;

vG0=[];
for i=1:k
 vG0=[vG0 (k-i)*ones(1,nt(i))];   
end;
G0=-inv(T)*diag(vG0)*T;

Z1=inv(B'*B)*B';
Z0=B*Z1;
At=A+B*K;


I_n=eye(n);I_m=eye(m);
P12=-0.5*B*inv(B'*B);

lambdas=real(eig(-P12*Ki-Ki'*P12'+A'*P12*P12'*A));
lam=max(0,-min(lambdas));
W1=[kron(I_n, At')+kron(At',I_n)];
Q1=(2+lam)*I_n+A'*P12*P12'*A-P12*Ki-Ki'*P12'; 
zet1=-Q1(:);
v_P1=linsolve(W1,zet1);
P=reshape(v_P1,n,n);  

if max(real(eig(P)))<tol disp('Error: The upgrade is impossible.'); return; end;

if max(real(eig(P*At+At'*P)))>-tol disp('Error: The upgrade is impossible.'); return; end;

W3=[kron(Ki',I_m)];
zet3=-B'*P;zet3=zet3(:);
W4=[kron(I_m,Ki')];
zet4=-P*B;zet4=zet4(:);
v_P2=linsolve([W3;W4],[zet3;zet4]);
P2=reshape(v_P2,m,m); 
 
P2=(P2+P2')/2;

if (max(real(eig(P2)))<tol)     disp('Error: The upgrade is impossible.'); return; end;

Ki_new=-inv(P2)*B'*P;

lambdas=real(eig(sqrtm(P)*G0*inv(sqrtm(P))+inv(sqrtm(P))*G0'*sqrtm(P)));
lambda_min=min(lambdas);
lambda_max=max(lambdas);
if lambda_max>tol mu_min=max(-1/2,-1/lambda_max+tol);
else mu_min=-1/2;
end;
if lambda_min<-tol mu_max=min(1/k, -1/lambda_min);
else mu_max=1/k;
end;



