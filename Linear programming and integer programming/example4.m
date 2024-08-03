c=[3;2];
a=[12,3;2,3;3,15];
b=[4;2;5];
aeq=[1,1];
beq=1;
x=linprog(c,-a,-b,aeq,beq,zeros(2,1))
value=c'*x