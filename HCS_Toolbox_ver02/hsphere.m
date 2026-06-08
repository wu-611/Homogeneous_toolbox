function M=hsphere(r,Gd,Nmax,hn_fun)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function M=hsphere(r, Gd, Nmax, hn_fun) returns a randomly generated an 
%% array of points distributed on a homogeneous sphere of the radius r
%%                        
%%  The homogeneous ball of the radius r>0 is the set of point 
%%
%%           Sd(r)={z : hz=r, where nz is a homogeneous norm of z}
%%
%%
%% The input parameters are                                   
%%     r - radius of the homogeneous sphere      
%%  Nmax - number of points on the shpere  
%% hn_fun - a function which computes a homogenenous norm 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tol=1e-6;
n=size(Gd,1);
if r<tol disp('Error: radius of homogenenous ball must be strctly positive'); end;

n=size(Gd,1);
s=log(r);
M=[];
for i=1:Nmax
    x=2*(rand(n,1)-0.5);
    if norm (x)<tol x=2*(rand(n,1)-0.5); end;
    M=[M expm(Gd*s)*hproj(x,Gd,hn_fun)];
end;   