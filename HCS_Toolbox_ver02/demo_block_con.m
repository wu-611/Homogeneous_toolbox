%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Demo of transformation to a block controlability form
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A= [ 8     8     8    10     7     8     3;
     7     8     3     3     9     3     6;
     3     2     7     6    10     8     5;
    10     5     7     2     5     2     4;
     0     4     2     8     1     9     8;
     4     6     1     3     1     3     6;
     4     7     5     5     3     2     5];

B=[  9     1;
     3     5;
     8     8;
     8     9;
     4     1;
     6     6;
     1     5];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Orthogonal transformation to a trangular block controllability form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[T nl]=trans_con(A,B)   % nl - sizes of blocks

A1=T*A*T'

B1=T*B

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Invertable transformation to a canonical block controllability form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M nl]=block_con(A,B);  % nl - sizes of blocks

A2=M*A*inv(M)

B2=M*B