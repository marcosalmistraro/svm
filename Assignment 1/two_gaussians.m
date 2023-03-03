% Displaying two Gaussians on 2D space

% Create random points according to two gaussians
X1 = randn(50, 2) + 1;
X2 = randn(51, 2) - 1;

% Create point labels
Y1 = ones(50, 1);
Y2 = -ones(51, 1);

% Display plot
figure;
hold on;
plot(X1(:,1), X1(:,2), 'ro');
plot(X2(:,1), X2(:,2), 'bo');
title('Two-class Gaussian distributions');
hold off;