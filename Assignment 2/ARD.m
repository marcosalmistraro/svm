%% Automatic Relevance Determination for feature selection
X = 6.*rand(100, 3) - 3;
Y = sinc(X(:,1)) + 0.1.*randn(100,1);
Xtrain=X;
Ytrain=Y;

% set prior values for sig2 and gam
sig2=0.01;
gam=1;

criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

% execute ARD ranking
[selected, ranking] = bay_lssvmARD({X, Y, 'f', gam, sig2});

rankscore = [3, 2, 1];
bar(ranking, rankscore)
yticks([1 2 3])
xlabel('Inputs')
ylabel('Relative ranking set by ARD')
title('ARD on Bayesian-optimized LSSVM')
