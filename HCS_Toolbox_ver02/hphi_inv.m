function x=hphi_inv(y,Gd,n_fun)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function x=hphi_inv(y,Gd,n_fun) computes (for a given vector y) 
%% a value of the inverse homogeneous homeomorphism on R^p:
%%                                                            
%%         x = Phi^{-1}(y) = d(ln||y||) * y / ||y||
%%
%% The input parameters are                                   
%%      y - vector (p x 1) 
%%     Gd - anti-Hurwitx matrix (p x p)  being the generator of
%%          the dilation d(s)=expm(s*Gd) for any real s 
%% n_fun  - a function which computes a norm of y (e.g., norm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if norm(y)==0 x=zeros(size(y));
else nm=n_fun(y); x=expm(log(nm)*Gd)*y/nm; 
end;
