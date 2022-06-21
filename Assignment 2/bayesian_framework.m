%% Bayesian framework
X = (-3:0.01:3)';
Y = sinc(X)+0.1.*randn(length(X),1);

Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);

sig2 = 0.025; gam = 158.4893; % select pre-tuned hyperparameters
criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

% compute error bars
sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');

Ypred = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'}, ...
    {alpha, b}, Xtest);
mse = mean((Ytest - Ypred).^2);
disp(mse)
