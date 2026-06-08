function [T nt]=trans_obs(A,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  The function [T n_list]=trans_con(A,C) computes           
%%  an orthogonal matrix T for an observable pair {A,C}:      
%%      T'*A*T= [ A11  A12  A13 ...    *       *     * ;                    
%%                A21  A22  A23 ...    *       *     * ;                    
%%                 0   A32  A33 ...
%%                ...  ...  ... ...   ...     ...   ...;                    
%%                 0   0    0   ... Ak-1k-2 Ak-1k-1  * ;      
%%                 0   0    0   ...    0     Akk-1  Akk]      
%%  and                                                       
%%                 C*T=[0 C0],                                
%%  where                                                     
%%    - Aij is a block of the size (ni x nj) with i,j=1,...,k   
%%    - C0  is a full-column-rank  matrix (m x n1)            
%%  The array n_list=[n1,...,nk] contains sizes of the blocks.  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T nt]=trans_con(A',C');
T=-T';