%% Robust regression on noisy sinc function
X = (-6:0.2:6)';
rng('default');
Y = sinc(X) + 0.1.*rand(size(X));

% adding outliers
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));

% robust model
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whampel'; % select weighting function for rubust statistics
model = tunelssvm(model, 'simplex', costFun, {10, 'mae';}, wFun);
model = robustlssvm(model);
plotlssvm(model);
