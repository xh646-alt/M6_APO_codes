function [sum_ret,stdev,IR] = performance2(p)

ret=log(1+p);

sum_ret=sum(ret);

stdev=std(ret);

IR=sum_ret/stdev;

end