function y=hdot(alpha,x,Gd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function y=hdot(alpha,x,Gd) computes a product of a scalar
%% alpha by a vector x in the homogeneous Euclidean space.
%%
%% The input parameters are                                   
%%      alpha - a real scalar                x - a vector (p x 1)  
%%      Gd - an anti-Hurwitx matrix (p x p) being the generator of
%%           the dilation d(s)=expm(s*Gd) for any real s 
%%      P  - positive definite matrix (p x p) such that P*Gd+Gd'*P>0 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if alpha==0 y=zeros(size(x));
else y=sign(alpha)*expm(log(abs(alpha))*Gd)*x;
end;


