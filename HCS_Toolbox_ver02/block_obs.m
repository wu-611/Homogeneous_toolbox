function [T nt]=block_obs(A,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  The function [T n_list]=block_obs(A,C) computes                 
%%  an invertible matrix T that transforms an observable pair {A,C} 
%%  to the canonical block observability form:                      
%%     inv(T)*A*T=[ 0   0   0  ...  0    A1k ;                      
%%                 A21  0   0  ...  0    A2k ; 
%%                  0  A23  0  ...  0    A3k ; 
%%                 ... ... ... ... ...   ... ;                             
%%                  0   0  ... ... Akk-1 Akk ]                             
%%  and                                                             
%%                 C*T=[0 C0],                                      
%%  where                                                           
%%     - Aij is a block of the size (ni x nj) with i,j=1,...,k      
%%     - C0 is a full-column-rank matrix (m x nk)                  
%%  The array n_list=[n1,...,nk] contains sizes of the blocks.      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T nt]=block_con(A',C');
T=-T';