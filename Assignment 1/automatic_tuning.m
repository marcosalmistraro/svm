% Simple script for showing automatic tuning on a LSSVM alogrithm

load iris

model = {X,Y,'c',[],[],'RBF_kernel','ds'}; % csa vs ds
[gam,sig2,cost] = tunelssvm(model,'simplex', ...
    'crossvalidatelssvm',{10,'misclass'}); 
disp(gam),
disp(sig2),
disp(cost)