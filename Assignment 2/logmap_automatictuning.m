clc;
clear;

load logmap.mat

order = 10;
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);

[gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel','csa',}, ...
    'simplex','crossvalidatelssvm', {10,'mae'});
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});

Xs = Z(end-order+1:end, 1);

nb = 50;
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);

mae = mean(abs(prediction-Ztest));

figure;
hold on;
plot(Ztest, 'k');
plot(prediction, 'r');
legend('Test Set', 'Predicted')
title(sprintf('Logmap dataset, gam=%s, sig2=%s, MAE=%s', num2str(gam), num2str(sig2), num2str(mae)));
hold off;


