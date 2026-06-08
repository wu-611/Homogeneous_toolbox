function [K0, K, G0, P, mu1, mu2, rho]=fhpc_design(A, B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [K0, K, G0, P, mu1, mu2, rho]=fhpc_design(A, B) computes                    
%% parameters of Fixed-time Homogeneous Proportional Controller (FHPC)     
%%          _ 
%%         /
%%        /  K0*x+nx^(1+mu1)*K*d1(-log(nx))*x       if nx<=1
%%   u = < 
%%        \  K0*x+nx^(1+mu2)*K*d2(-log(nx))*x       if nx>1
%%         \_  
%%      
%%                    (nx - homogeneous norm of x)         
%%                                                                      
%% which stabilizes the linear MIMO system in a fixed-time  
%%             
%%        |x(t)|=0, t>=Tmax, where Tmax=1/(-mu1*rho) + 1/(mu2*rho);
%%                                                                      
%%                                                                      
%% The input parameters:                                             
%%     A  - system matrix (n x n)                                       
%%     B  - control matrix (n x m)               
%%                                                                      
%% The output parameters:                                                
%%     K0 - feedback gain (m x n) of the linear term of FHPC          
%%     K  - feedback gain (m x n) of the nonlinear term of FHPC       
%%     G0 -  matrix (n x n) defines generators of dilations d1(s) and d2(s) 
%%              Gd1=eye(n)+mu1*G0 and Gd2=eye(n)+mu2*G0                   
%%     P  - positive definite matrix which defines the weighted Euclidean 
%%         norm sqrt(x'*P*x) that induces the canonical homogeneous norm nx  
%%     mu1 - minimal admissible negative homogeneity degree 
%%     mu2 - maximal admissible positive homogeney degree 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K0=0; K=0; G0=0; P=0; mu1=0; mu2=0; rho=0;

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
    if rank(U)==n k=i; end;
    tA=A*tA;
end;
if k==0 disp('Error: The system is not controllable'); return;
end;

if det(U*U')<tol  disp('Warning: The system is weakly controllable. Parameters may be badly tuned due to computations issues.');
end;    


if rank(B)==n 
   K0=-B'*inv(B*B')*A;
   K=-B'*inv(B*B'); Gd=eye(n);
   P=eye(n);Gd=P; mu1=-1;mu2=1; 
   rho=2;
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
 vG0=[vG0 (k-i)*ones(1,nt(i))];   
end;
G0=-diag(vG0);
Gd=eye(n);R=eye(n);

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
X=inv(T)*X*inv(T)';
P=inv(X);
K=K*T;
G0=inv(T)*G0*T;
lambda_min=min(real(eig(sqrtm(P)*G0*inv(sqrtm(P))+inv(sqrtm(P))*G0'*sqrtm(P))));
lambda_max=max(real(eig(sqrtm(P)*G0*inv(sqrtm(P))+inv(sqrtm(P))*G0'*sqrtm(P))));
if lambda_max>tol mu1=max(-1,-1/lambda_max+tol);
else mu1=-1;
end;
if lambda_min<-tol mu2=min(1/k, -1/lambda_min-tol);
else mu2=1/k;
end;

delta=max(mu2*lambda_max,mu1*lambda_min);
if delta<0 rho=1;
else rho=(2+min(real(eig(X))))/(2+delta);
end; 

catch
 K0=0; K=0;P=0; Gd=0; disp('Error: The control gains cannot be found with a block decomposition method');
end;

