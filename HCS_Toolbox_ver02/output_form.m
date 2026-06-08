function [tK, As, Bs, err]=output_form(A,B,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [tK, As, Bs]=output_form(A,B,C) computes  the linear                 
%% feedback gain tK allowing a reduction of a dynamics of the output 
%%
%%                             y=C*x 
%% 
%%  of the linear control system 
%%
%%                        dx/dt=A*x+B*u
%% 
%%  with the control law 
%% 
%%                         u=tK*x+v    
%%    
%% to the form
%%
%%                     dy/dt=As*y+Bs*v
%%
%% where x - system state (n x 1),     A - system matrix (n x n), 
%%       B - control matrix (n x m),   u - control input (m x 1), 
%%       C - output matrix (p x n),    y - system output (p x 1),  
%%       tK - matrix of linear feedback gains (m x n),
%%       v -  new control input (m x n),
%% 
%% Input parameters: A,B,C  Output parameters: tK, As, Bs
%%
%% A warning or an error will appear if such a reduction is impossible.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

As=0;Bs=0;tK=0;err=0;
if nargin<3 disp('Error: not enough input arguments'); err=1; return; end;

%1) Check of parameters
sA=size(A); sB=size(B); sC=size(C);

if sA(1)~=sA(2) disp('Error: the system matrix must be square'); err=2;return;end;
if sB(1)~=sA(2) disp('Error: dimensions of system and control matrices must agree');err=3;return;end;
if sA(2)~=sC(2) disp('Error: dimensions of system and output matricies must agree'); err=4;return;end;
if sC(1)>sA(1)  disp('Error: output matrix  is too large'); return;err=5;end;
if sC(1)~=rank(C) disp('Error: output matrix must have full row rank'); err=6;return;end;
%check surface
tol=0.00001;
n=size(A,2); p=size(C,1); m=size(B,2);
Prj=eye(n)-C'*inv(C*C')*C;
W=kron(Prj',C*B);
tmpA=-C*A*Prj;
zet=tmpA(:);
[v_tK Res]=linsolve(W,zet);
tK=reshape(v_tK,m,n);

if norm(C*(A+B*tK)*Prj)>tol
    disp('Warning: the reduction seems to impossible. The obtained output model may be controllable.'); err=-1;
end;
As=C*(A+B*tK)*C'*inv(C*C');
Bs=C*B;
