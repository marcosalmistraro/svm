%% Regression of the sinc function using LSSVM and automated tuning
X = (-3:0.01:3)';
Y = sinc(X)+0.1.*randn(length(X),1);

Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);

% using tunelssvm
optFun = 'simplex'; % gridsearch or simplex
globalOptFun = 'csa'; % csa or ds

tic
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', ... 
    globalOptFun},optFun,'crossvalidatelssvm',{10,'mse'})
toc

disp(gam)
disp(sig2)
disp(cost)

% evaluation
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
%plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});

Ypred = simlssvm({Xtrain,Ytrain,'f',gam,sig2, ...
    'RBF_kernel'},{alpha,b},Xtest);

plot(Xtest,Ytest,'.', 'MarkerSize', 15);
hold on;
plot(Xtest,Ypred,'r-+', 'LineWidth', 1);
legend('Ytest','Ypred');

mse = mean((Ytest - Ypred).^2);
disp(mse)
