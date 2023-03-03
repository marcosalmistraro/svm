%% Fixed-size LSSVM script

clear
close all

X = 3.*randn(100,2);
ssize = 10;
sig2 = 1;
subset = zeros(ssize,2);
for t = 1:100,

  % New candidate subset
  r = ceil(rand*ssize);
  candidate = [subset([1:r-1 r+1:end],:); X(t,:)];
  
  % Evaluate whether current candidate is better than the previous
  if kentropy(candidate, 'RBF_kernel',sig2)>...
        kentropy(subset, 'RBF_kernel',sig2),
    subset = candidate;
  end
  
 % Display figure 
  plot(X(:,1),X(:,2),'b*'); hold on;
  plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
  pause(1)

end