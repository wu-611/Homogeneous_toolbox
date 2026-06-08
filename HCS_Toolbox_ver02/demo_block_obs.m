%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Demo of transformation to a block observability form
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A= [ 8     8     8    10     7     8     3;
     7     8     3     3     9     3     6;
     3     2     7     6    10     8     5;
    10     5     7     2     5     2     4;
     0     4     2     8     1     9     8;
     4     6     1     3     1     3     6;
     4     7     5     5     3     2     5];

C= [9     3     8     8     4     6     1;
    1     5     8     9     1     6     5];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Orthogonal transformation to a trangular block observability form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[T nl]=trans_obs(A,C)   % nl - sizes of blocks

A1=T'*A*T

C1=C*T

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Invertable transformation to a canonical block observability form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M nl]=block_obs(A,C);  % nl - sizes of blocks

A2=inv(M)*A*M

C2=C*M