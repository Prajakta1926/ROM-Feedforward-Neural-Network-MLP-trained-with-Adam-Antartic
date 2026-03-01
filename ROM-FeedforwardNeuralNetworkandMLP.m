clear; clc; close all;

%% 1. DATA
N = 20000;
t = linspace(0,50,N)';

TEMP_clean = -10.23 + 4.67*sin(2*pi*t/24) + 2.5*sin(2*pi*t/12);
WS_clean   = 9.84 + 3.12*cos(2*pi*t/24) + 1.8*sin(2*pi*t/6);

TEMP = TEMP_clean + 0.001*randn(N,1);
WS   = WS_clean   + 0.001*randn(N,1);

X = [TEMP WS];

%% 2. NORMALIZE
X_mean = mean(X,1);
X_std  = std(X,0,1);
X_norm = (X - X_mean) ./ X_std;

%% 3. ROM (SVD)
[U,S,~] = svd(X_norm,'econ');
sing_vals = diag(S);
energy = cumsum(sing_vals.^2)/sum(sing_vals.^2)*100;

K = find(energy>=99.99,1);
U = U(:,1:K);

fprintf("K = %d (%.4f%% energy)\n",K,energy(K));

%% 4. 80:20 SPLIT
total = size(U,1);
train_n = round(0.8*total);

U_train = U(1:train_n,:);
X_train = X_norm(1:train_n,:);

U_test  = U(train_n+1:end,:);
X_test  = X_norm(train_n+1:end,:);

train_samples = size(U_train,1);

%% 5. NETWORK (XAVIER INIT)
W1 = randn(K,128)*sqrt(2/(K+128)); b1 = zeros(1,128);
W2 = randn(128,64)*sqrt(2/(128+64)); b2 = zeros(1,64);
W3 = randn(64,2)*sqrt(2/(64+2)); b3 = zeros(1,2);

%% Adam
lr = 0.001;
beta1=0.9; beta2=0.999; eps=1e-8;

mW1=0; vW1=0; mW2=0; vW2=0; mW3=0; vW3=0;
mb1=0; vb1=0; mb2=0; vb2=0; mb3=0; vb3=0;
t_adam=0;

epochs = 10000;
batch_size = 256;

fprintf("Training %d epochs...\n",epochs);

%% 6. TRAINING
for epoch = 1:epochs
    
    idx = randperm(train_samples);
    
    for batch = 1:batch_size:train_samples
        
        batch_end = min(batch+batch_size-1,train_samples);
        batch_idx = idx(batch:batch_end);
        
        Ub = U_train(batch_idx,:);
        Xb = X_train(batch_idx,:);
        current_batch = size(Ub,1);
        
        % Forward
        a1 = tanh(Ub*W1 + b1);
        a2 = tanh(a1*W2 + b2);
        y  = a2*W3 + b3;
        
        % Gradient
        dy = 2*(y - Xb)/current_batch;
        
        dW3 = a2'*dy; db3 = sum(dy,1);
        da2 = dy*W3';
        dz2 = da2.*(1-a2.^2);
        
        dW2 = a1'*dz2; db2 = sum(dz2,1);
        da1 = dz2*W2';
        dz1 = da1.*(1-a1.^2);
        
        dW1 = Ub'*dz1; db1 = sum(dz1,1);
        
        % Adam updates
        t_adam = t_adam+1;
        
        [W1,mW1,vW1]=adam(W1,dW1,mW1,vW1,lr,beta1,beta2,eps,t_adam);
        [b1,mb1,vb1]=adam(b1,db1,mb1,vb1,lr,beta1,beta2,eps,t_adam);
        
        [W2,mW2,vW2]=adam(W2,dW2,mW2,vW2,lr,beta1,beta2,eps,t_adam);
        [b2,mb2,vb2]=adam(b2,db2,mb2,vb2,lr,beta1,beta2,eps,t_adam);
        
        [W3,mW3,vW3]=adam(W3,dW3,mW3,vW3,lr,beta1,beta2,eps,t_adam);
        [b3,mb3,vb3]=adam(b3,db3,mb3,vb3,lr,beta1,beta2,eps,t_adam);
        
    end
    
    % Learning rate decay
    if epoch==3000, lr=0.0005; end
    if epoch==6000, lr=0.0002; end
    if epoch==8000, lr=0.0001; end
    
    if mod(epoch,1000)==0
        fprintf("Epoch %d done\n",epoch);
    end
    
end

%% 7. TEST
a1 = tanh(U_test*W1 + b1);
a2 = tanh(a1*W2 + b2);
test_pred_norm = a2*W3 + b3;

test_pred = test_pred_norm .* X_std + X_mean;
X_test_original = X_test .* X_std + X_mean;

rmse = sqrt(mean((X_test_original - test_pred).^2,'all'));

fprintf("\nFINAL RMSE = %.8f\n",rmse);

%% Adam function
function [param,m,v]=adam(param,grad,m,v,lr,b1,b2,eps,t)
m=b1*m+(1-b1)*grad;
v=b2*v+(1-b2)*(grad.^2);
mhat=m/(1-b1^t);
vhat=v/(1-b2^t);
param=param-lr*mhat./(sqrt(vhat)+eps);
end