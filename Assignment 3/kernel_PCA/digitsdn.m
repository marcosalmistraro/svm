%
% Experiments on the handwriting data set on kPCA 
% for reconstruction and denoising
%

load digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));

% Add noise to the digit maps

noisefactor =0.3;

noise = noisefactor*maxx; % sd for Gaussian noise

Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest1; 
for i=1:size(Xtest1,1);
  randn('state', N+i);
  Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

% Select training set

Xtr = X(1:1:end,:);
sig2 =dim*mean(var(Xtr)); % rule of thumb
sigmafactor = 0.7;
sig2=sig2*sigmafactor;

% Kernel-based Principal Component Analysis 
% using the original training data

disp('Kernel PCA: extract the principal eigenvectors in feature space');
disp(['sig2 = ', num2str(sig2)]);

% Linear PCA
[lam_lin,U_lin] = pca(Xtr);

% Kernel PCA
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

% Denoise using the first principal components

disp('Denoise using the first PCs');

% Choose the digits for the test

digs=[0:9]; ndig=length(digs);
m=2; % Choose the m-th datapoint for each digit 

Xdt=zeros(ndig,dim);

% Displaying all digits

figure; 
colormap('gray'); 
title('Denosing using linear PCA'); tic

% Select number of eigenvalues for kpca
npcs = [2.^(0:7) 190];
lpcs = length(npcs);

for k=1:lpcs;
 nb_pcs=npcs(k); 
 disp(['nb_pcs = ', num2str(nb_pcs)]); 
 Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);   
 for i=1:ndig
   dig=digs(i);
   fprintf('digit %d : ', dig)
   xt=Xnt(i,:);
   if k==1 
     % plot the original clean digits
     subplot(2+lpcs, ndig, i);
     pcolor(1:15,16:-1:1,reshape(Xtest1(i,:), 15, 16)'); shading interp; 
     set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
     if i==1, ylabel('original'), end 
     % plot the noisy digits 
     subplot(2+lpcs, ndig, i+ndig); 
     pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
     set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
     if i==1, ylabel('noisy'), end
     drawnow
   end    
   Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
   subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
   pcolor(1:15,16:-1:1,reshape(Xdt(i,:), 15, 16)'); shading interp; 
   set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);           
   if i==1, ylabel(['n=',num2str(nb_pcs)]); end
   drawnow    
 end
end

% Denosing using Linear PCA for comparison

% Select number of eigenvalues for pca
npcs = [2.^(0:7) 190];
lpcs = length(npcs);

figure; colormap('gray');title('Denosing using linear PCA');

for k=1:lpcs;
 nb_pcs=npcs(k); 
 Ud=U_lin(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
 for i=1:ndig
    dig=digs(i);
    xt=Xnt(i,:);
    proj_lin=xt*Ud; % projections of linear PCA
    if k==1 
        % plot the original clean digits
        subplot(2+lpcs, ndig, i);
        pcolor(1:15,16:-1:1,reshape(Xtest1(i,:), 15, 16)'); shading interp; 
        set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);                
        if i==1, ylabel('original'), end  
        % plot the noisy digits 
        subplot(2+lpcs, ndig, i+ndig); 
        pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
        set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
        if i==1, ylabel('noisy'), end
    end
    Xdt_lin(i,:) = proj_lin*Ud';
    subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
    pcolor(1:15,16:-1:1,reshape(Xdt_lin(i,:), 15, 16)'); shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
    if i==1, ylabel(['n=',num2str(nb_pcs)]), end
 end
end