clear; clc

vl_ds = csvread('datasets/dataset_validation.csv');

hl_array = 4:2:16;
nb_nn = length(hl_array);

px_e = [0,4,8,12,16,20,24,32];
nb_ds = length(px_e);

cc_array = 1:1:20;
nb_copies = length(cc_array);

error = ones(nb_ds, nb_nn, nb_copies)*Inf;
for cc_ix=cc_array
    for ds_ix=1:nb_ds
        filename = sprintf('datasets/dataset_noise_%dpx_%d.csv',px_e(ds_ix), cc_ix);    
        fprintf("\n\n%s\n", filename);
        tr_ds = csvread(filename);

        n = length(tr_ds);
        idx = randperm(n) ;
        tr_ds_shuffled = tr_ds;
        tr_ds_shuffled(idx,:) = tr_ds;
        tr_ds = tr_ds_shuffled;

        % tr_ds = tr_ds(1:100,:);


        tr_x = transpose(tr_ds(:,4:9));% (px,py) x340
        tr_t = transpose(tr_ds(:,1:3)); % Object pose in 3D

        vl_x = transpose(vl_ds(:,4:9));% (px,py) x340
        vl_t = transpose(vl_ds(:,1:3)); % Object pose in 3D


        %% Construct a feedforward network with one hidden layer of size 'hl_array(hl_ix)'
        for hl_ix=1:nb_nn
           fprintf('%d hidden layers\n', hl_array(hl_ix));

           net = feedforwardnet(hl_array(hl_ix), 'trainbr');
           net.trainParam.Epochs = 20;
           %'trainlm', 'trainbr', 'trainbfg', 'trainrp', 'trainscg', 'traincgb' ,
           %'traincgf' ,'traincgp' ,'trainoss' ,'traingdx' ,'traingdm' ,'traingd'

           %% Train the network net using the training data.
           net = train(net,tr_x,tr_t);

           %% Assess the performance of the trained network
           diff=abs(vl_t-net(vl_x));
           e_mm = sqrt(mean(diff(1,:))^2 + mean(diff(2,:))^2 + mean(diff(3,:))^2)*1000;
           fprintf('Error in [mm]: %3.3f\n', e_mm);



           net_name = sprintf("mlp_%dpx_%dhn", px_e(ds_ix), hl_array(hl_ix));
           error(ds_ix, hl_ix, cc_ix) = e_mm;
           save('mlps/'+net_name, 'net');


        end %% After Training MLP

    end %% After Reading *.csv
end

datetime.setDefaultFormats('default','yyyy_MM_dd_hh_mm')
save('ErrorAll_'+string(datetime), 'error');



