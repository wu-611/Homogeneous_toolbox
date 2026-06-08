function y=lp_norm(tl,xl,a,b,p)
y=0;
if a>b disp('Error: a must be less than b'); return;
elseif a==b return;
else
    i=1;j=size(tl,2);
    if tl(1)>a a=tl(1); end;
    if b>tl(end) b=tl; end;
    while (tl(i)<a) && (i<j) i=i+1; end;
    while (tl(j)>b) && (j>1) j=j-1; end; 
    if i<j
       if p==Inf
          y=max(max(abs(xl(:,i:j))));
       elseif p>=1
          s=0;
          for k=i:j-1
            s=s+(tl(i+1)-tl(i))*0.5*sum(abs(xl(:,i)).^p+abs(xl(:,i+1)).^p);
          end;
          y=s^(1/p);
       end
     end;
end;
