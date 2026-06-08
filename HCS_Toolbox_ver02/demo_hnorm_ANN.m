%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Example of approximation of the canonical (implict) homogeneous norm by
%% a homogenenous artifical neural network (ANN) 
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol=1e-6;                                                                  % compuational tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Shape matrix of the wighted Euclidean norm
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P=[0.1157   0.0194;
   0.0194   0.1186];                                                             

if norm(P-P')>tol disp('Error: shape matrix P must be symmetric'); return; end;

if min(real(eig(P)))<tol disp('Error: shape matrix P must be positive definite'); return; end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Generator of the linear dilation in R^2
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gd=[5 0; 0 4]; 

[J L]=eig(Gd);

L=diag(L); J=inv(J);                                                       % diagonalization of the generator
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Approximation of the canonical homogeneous norm by ANN
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rb=5;                                                                      % smoothing parameter

Q=approx_hnorm_r(Gd,P,rb,50);                                              % optimal selection of weight Q (needs YALMIP)

[alpha beta gamma]=approx_hnorm_ANN(Gd,P,@(x)hnorm_r(x,L,rb,Q,J),10);      %approximation by ANN
                                                                 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Computation of homogeneous norm 
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
%plotting the unit homogeneous spheres

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');



N=5000; %number of points on the sphere
M=hsphere(1,Gd,N,@(x)hnorm(x,Gd,P));
plot1 = plot(M(1,:),M(2,:),'Marker','.','LineStyle','none');
set(plot1(1),'DisplayName','canonical homogeneous norm $\|x\|_{\mathbf{d}}$',...
    'MarkerFaceColor',[1 0 0],...
    'Color',[1 0 0]);


M=hsphere(1,Gd,N,@(x)hnorm_r(x,L,rb,Q,J));
plot2 = plot(M(1,:),M(2,:),'Marker','.','LineStyle','none');
set(plot2(1),'DisplayName','explicit homogeneous norm $\|x\|_r$',...
    'MarkerFaceColor',[0 0 1],...
    'Color',[0 0 1]);

M=hsphere(1,Gd,N,@(x)hnorm_ANN(x,Gd,alpha,beta,gamma,@(x)hnorm_r(x,L,rb,Q,J)));
plot2 = plot(M(1,:),M(2,:),'Marker','.','LineStyle','none');
set(plot2(1),'DisplayName','explicit homogeneous norm $\|x\|_{\mathbf{d},\epsilon}$ by ANN',...
    'MarkerFaceColor',[0 1 0],...
    'Color',[0 1 0]);



xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
 xlim(axes1,[-5 5]);
 ylim(axes1,[-5 5]);
box(axes1,'on');
set(axes1,'FontSize',30,'XGrid','on','YGrid','on');

legend1 = legend(axes1,'show');
set(legend1,'Interpreter','latex');

err=0;
for i=1:N
  err=max(err, abs(hnorm(M(:,i),Gd,P)- hnorm_ANN(M(:,i),Gd,alpha,beta,gamma,@(x)hnorm_r(x,L,rb,Q,J)))); 
end;
err
