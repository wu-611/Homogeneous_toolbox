function y=hadd(x1,x2,Gd,P)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function y=hadd(x1,x2,Gd,P) computes sum of vectors x1 and x2
%% in the homogeneous Euclidean space.
%%
%% The input parameters are                                   
%%      x1 - a vector (p x 1)               x2 - a vector (p x 1)  
%%      Gd - an anti-Hurwitx matrix (p x p) being the generator of
%%           the dilation d(s)=expm(s*Gd) for any real s 
%%      P  - positive definite matrix (p x p) such that P*Gd+Gd'*P>0 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hn_fun=@(x)hnorm(x,Gd,P);
y=hphi_inv(hphi(x1,Gd,hn_fun)+hphi(x2,Gd,hn_fun),Gd,@(x)sqrt(x'*P*x));


