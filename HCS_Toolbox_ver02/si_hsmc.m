function u=si_hsmc(h, x, A, B, C, K0, K, Gd, P, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function  u=si_hsmc(h, x, A, B, C, K0, K, Gd, P) discretizes                
%% semi-implicitly Homogeneous (unit) Sliding Mode Controller (HSMC)     
%%                                                                      
%%     u = K0*x + K*d(-ln(ncx))*x 
%%
%%  where x - system state  and   ncx - homogenenous norm of C*x
%%
%%
%% The function  u=si_hsmc(h, x, A, B, C, K0, K, Gd, P, alpha, beta)                    
%% uses the saturation parameters alpha and beta to lower/upper bound
%% the homogenenous norm nx     
%%                                                                      
%%     ncx = max(alpha, min(beta,ncx)) 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin==11 hn=hnorm(C*x,Gd,P,alpha,beta);
   elseif nargin==10 hn=hnorm(C*x,Gd,P,alpha);
   else hn=hnorm(C*x,Gd,P); 
end;
u=zeros(size(K0,1),1);
n=size(A,2);I_n=eye(n);
if hn>1e-20 %numerical tolerance
Kh=K0+K*expm(-log(hn)*Gd)*C;
u=Kh*inv(I_n-h*B*Kh)*(I_n+h*A)*x;
end;


