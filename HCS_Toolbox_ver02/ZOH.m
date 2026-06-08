function [Ah,Bh]=ZOH(h,A,B,tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [Ah,Bh]=ZOH(h,A,B) computes Zero-Order-Hold discretization
%% 
%%   x(k+1)=Ah*x+Bh*u
%%
%%   of continuous-time linear system
%% 
%%   dx/dt=A*x+B*u
%%
%%  The input parameters:
%%  
%%     h  - sampling period
%%     A  - system matrix
%%     B  - control matrix
%%
%% The function [Ah,Bh]=ZOH(h,A,B,tol) uses additionally the parameter tol
%% to control computation precision.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<4 tol=1e-16;end;


Ai=h*eye(size(A,1)); 
S=0;i=1;
while (norm(Ai)>tol)
S=S+Ai;i=i+1;
Ai=Ai*A*h/i;
end;
Bh=S*B;
Ah=eye(size(A,1))+A*S;
IA=S;



