clear; clc

ds = csvread('dataset.csv');
ix_split = round(length(ds)*0.7);

tr_x = transpose(ds(1:ix_split,4:9));% (px,py) x340
tr_t = transpose(ds(1:ix_split,1:3)); % Object pose in 3D

vl_x = transpose(ds(ix_split+1:length(ds),4:9));% (px,py) x340
vl_t = transpose(ds(ix_split+1:length(ds),1:3)); % Object pose in 3D


%% Construct a feedforward network with one hidden layer of size 'nbl'
for nbl = [2,3,4,6,8,12,16]
   fprintf('%d hidden layers\n', nbl);
   
   net = feedforwardnet(nbl, 'trainbr');
   %'trainlm', 'trainbr', 'trainbfg', 'trainrp', 'trainscg', 'traincgb' ,
   %'traincgf' ,'traincgp' ,'trainoss' ,'traingdx' ,'traingdm' ,'traingd'

   %% Train the network net using the training data.
   net = train(net,tr_x,tr_t);

   %% Estimate the targets using the trained network.
   y = net(vl_x);

   %% Assess the performance of the trained network
   perf = perform(net,y,vl_t);
   fprintf('MSE(%d hns): %3.12f\n', nbl, perf);

   net_name = sprintf("mlp_%d", nbl);
   save(net_name, 'net')
    
end