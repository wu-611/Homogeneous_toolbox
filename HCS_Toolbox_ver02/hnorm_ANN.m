function nx=hnorm_ANN(x,Gd,alpha,beta,gamma,hn_fun)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function nx=hnorm_ANN(x, Gd, alpha, beta, gamma, hn_fun) 
%% computes (for a given vector x) a homogeneous norm defined by 
%% Artifical Neural Network (ANN) 
%% 
%%
%% The input parameters are
%%
%%    x      - vector (p x 1) which norm has to be computed  
%%    Gd     - anti-Hurwitx matrix (p x p)  being the generator of
%%             the dilation d(s)=expm(s*Gd) for any real s 
%%    alpha  - (N x n) \
%%    beta   - (N x 1)  >  parameters of ANN tunned by  approx_hnorm_ANN
%%    gamma  - (1 x N) / 
%%    hn_fun - a function which computes an explicit homogeneous norm 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nz=hn_fun(x); pi_z=expm(-log(nz)*Gd)*x;
nx=0;
for i=1:size(alpha,2)
nx=nx+gamma(1,i)/(1+exp(alpha(1,i)+beta(:,i)'*pi_z));
end;
nx=nx*nz;

