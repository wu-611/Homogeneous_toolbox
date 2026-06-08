function C=hcone(H,Gd,P,D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function C=hcone(H,Gd,P,D) generates points on the unit sphere
%% beloning to a linear d-homogeneous cone
%%                                                            
%% The input parameters are                                   
%%     H  -  (m x p) matrix of linear cone  
%%    Gd  - anti-Hurwitx matrix (p x p)  being the generator of
%%          the dilation d(s)=expm(s*Gd) for any real s 
%%      P - ashape matrix of the unit sphere x'*P*x=1
%%      D - a density (number of points per the unit sphere)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C=[];n=size(P,1); iP12=sqrtm(inv(P));

for i=1:round(D)
    V=(rand(n,1)-0.5)*2;
    V=iP12*V/norm(V,2);
    if min(H*V)>=0 C=[C V]; end;
end;
