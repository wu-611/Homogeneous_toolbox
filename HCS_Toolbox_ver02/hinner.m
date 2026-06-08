function q=hinner(x,y,Gd,P)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function q=(x,y,Gd,P) computes an inner prodcut of vectors x and y
%% in the homogeneous Euclidean space.
%%
%% The input parameters are                                   
%%      x - a vector (p x 1)              y - a vector (p x 1)  
%%      Gd - an anti-Hurwitx matrix (p x p) being the generator of
%%           the dilation d(s)=expm(s*Gd) for any real s 
%%      P  - positive definite matrix (p x p) such that P*Gd+Gd'*P>0 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if     norm(x)==0 q=0;
elseif norm(y)==0 q=0;
else   hn_fun=@(z)hnorm(z,Gd,P); 
       q=hphi(x,Gd,hn_fun)'*P*hphi(y,Gd,hn_fun);
end;


