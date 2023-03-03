%% Regression of the sinc function using LSSVM

X = (-3:0.01:3)';
Y = sinc(X)+0.1.*randn(length(X),1);

Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);

% Modify the following parameters to evaluate effects
gam = 100; % 10, 100
sig2 = 5; % 0.1, 1
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});

YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, ...
    {alpha,b},Xtest);

plot(Xtest,Ytest,'.', 'MarkerSize', 15);
hold on;
plot(Xtest,YtestEst,'r-+', 'LineWidth', 1);
legend('Real values','Estimation');
title('Sinc function regression, gam=100, sig2=5');

% Plot MSE on the test set
mse = mean((Ytest - YtestEst).^2);
disp(mse)