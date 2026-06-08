function y=hphi(x,Gd,hn_fun)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function y=hphi(x,Gd,hn_fun) computes (for a given vector x) a value of               
%% the homogeneous homeomorphism on R^p:
%%                                                            
%%         y = Phi(x) = ||x||_d * d(- ln||x||_d) * x
%%
%% The input parameters are                                   
%%       x - a vector (p x 1)   
%%      Gd - an anti-Hurwitx matrix (p x p) being the generator of
%%           the dilation d(s)=expm(s*Gd) for any real s 
%% hn_fun  - a function which computes homogeneous norm of x (e.g., hnorm) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if norm(x)==0 y=zeros(size(x));
else hn=hn_fun(x); y=hn*expm(-log(hn)*Gd)*x; 
end;
