function u=c_hpc(h, tn, x, A, B, K0, K, Gd, mu, rho, hn_fun)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function  u=c_hpc(h, tn, x, A, B, K0, K, Gd, P, mu, rho, hn_fun)   
%% discretizes consistently (*) Homogeneous Proportional Controller (HPC)     
%%                                                                      
%%     u = K0*x + nx^(1+mu)*K*d(-ln(nx))*x 
%%
%%  where x - system state  and   nx=hn_fun(x) - homogenenous norm of x
%%
%% (*) Consistent discretization preserves finite/fixed-time convergence 
%% of the continuous-time system
%% 
%%    dx/dt = A*x + B*u
%%  
%%  in its descrete-time counterpart
%% 
%%      x(k+1)=Ah*x(k)+B_h*u(k)
%%
%% where Ah=expm(h*A) and Bh=int^h_0  expm(s*A)*B ds
%%   
%% Important: The consistency of the discretization is theoretically proven
%% only for SISO system with nilpotent system matrix A (n x n). However, it
%% is defined for MIMO system (without proof) and can be computed by c_hpc.
%%
%% The natural number tn is such that 
%%
%%     rank [B, A*B,...,A^(tn-1)B] = n 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mu==0 disp('Error: homogeneity degree must be nonzero'); return; end;
  
hn=hn_fun(x); 

if hn^(-mu)>-mu*rho*h*tn
F0=A+B*(K0+K)+rho*Gd;
s=log(1+mu*rho*h*tn*hn^mu)/(rho*mu);
Q=expm(Gd*log(hn))*expm(-rho*Gd*s)*expm(F0*s)*expm(-Gd*log(hn));
else
    Q=zeros(size(A));
end;

[Ah,Bh]=ZOH(h,A,B);

W=[Bh];R=Bh;
for i=1:tn-1
R=Ah*R;
W=[W R];
end;
%u=R'*inv(W*W')*(Q-Ah^tn)*x;
Ul=W\((Q-Ah^tn)*x);
u=Ul(end-size(B,2)+1:end,1);










