function zl=hcurve(x,Gd,sl)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function zl=hcurve(x, Gd, sl) generates an array zl of points                   
%% of a homogenenous curve crossing the point x.
%%                        
%%  The homogenenous curve is the set given by 
%%
%%           Gamma = {z :  z=dn(s)x, -Inf<s<Inf}
%%
%% where d(s)=expm(Gd*s) is a dilation in R^p 
%%  
%% The input parameters are                                   
%%     x - vector (p x 1)   
%%    Gd - anti-Hurwitx matrix (p x p)  being the generator of
%%          the dilation d(s)=expm(s*Gd) for any real s 
%%    sl - array of scalars s for generation of zl 
%%
%%  The output parameter 
%%    zl - array of points on homogeneous cureve corresponding to sl 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tol=1e-6;

zl=[];
for i=1:max(size(sl))
zl=[zl expm(Gd*sl(i))*x];
end;