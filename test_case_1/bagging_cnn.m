%% bagging cnn ensemble method

%% ihepc dataset ----------------------------------------------------------
clear all
load ..\ihepc_dataset.mat
%% divide the training, validation, and testing dataset
start_index = 37+360 + 15*24*60;  % time(start_index) is 01-Jan-2007 00:00:00
end_index = start_index + (31+28)*24*60; 

tr_index = [start_index : end_index]; 
va_index = [end_index: end_index + 7*24*60];
te_index = [end_index + 7*24*60 : end_index + 7*24*60+7*24*60];

time = time(1: te_index(end)+1); active_power = active_power(1:te_index(end)+1);

%% generate the training data_set and validation data_set and test data_set
% for cnn model input (input size equals 200)
% training data_set
% normalize 
[tr_dataset, tr_min, tr_gap] = minmaxnor(active_power(1 : tr_index(end), :));
k = 1;
for i = tr_index(1) : tr_index(end) - 200
    tr_x{k} = tr_dataset(i: i + 200-1, :)';
    tr_y{k} = tr_dataset(i + 200, 1)';
    k = k+1;
end
% construct the 4-D double  training dataset
for i = 1: size(tr_x, 2)
    tr_X(:, :, 1, i) = tr_x{i};
    tr_Y(i,:) = tr_y{i}; 
end

for i = 1:5
% bagging the training dataset with 0.8
r_index = randperm(size(tr_X,4));
train_data = tr_X(:,:,:,r_index(1:floor(size(r_index, 2)*0.8)));
train_y = tr_Y(r_index(1:floor(size(r_index, 2)*0.8)));

% construct the CNN model
layers = [
    imageInputLayer([1 200 1])
    convolution2dLayer([1 8],8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer([1,2])
    convolution2dLayer([1 4],16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MiniBatchSize',300, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Plots','none', ...
    'ExecutionEnvironment', 'gpu',...
    'Verbose',false);

[cnn_net{i}, info] = trainNetwork(train_data,train_y,layers,options);
training_loss{i} = info.TrainingRMSE;
end
save('case1_ihepc_bagging_cnn_model.mat', 'cnn_net')

% testing data_set
load case1_ihepc_bagging_cnn_model.mat
[te_dataset(te_index(1)-200: te_index(end)), te_min, te_gap] ...
        = minmaxnor(active_power(te_index(1)-200:te_index(end), :));
te_dataset = te_dataset';
k = 1;
for i = te_index(1): te_index(end)
    te_x{k} = te_dataset(i-(200): i-1, :)';
    te_y{k} = te_dataset(i, 1)';
    k = k+1;
end
% construct the 4-D doubule 
for i = 1: size(te_x, 2)
    te_X(:, :, 1, i) = te_x{i};
    te_Y(i,:) = te_y{i}; 
end

for j = 1:5
    for i = 1: size(te_x, 2)
        YPre(j,i) = predict(cnn_net{j},te_X(:, :, 1, i), 'ExecutionEnvironment','gpu');
    end
end
Ypre1 = mean(YPre, 1); % the bagging forecasting
% min max reverse
bagging_cnn_pre = versenor(Ypre1, te_min, te_gap);
save('case1_ihepc_bagging_cnn_forecast.mat', 'bagging_cnn_pre')

te_value = active_power(te_index)';
% rmse
rmse_ihepc_bagging_cnn = rmse_1(te_value, bagging_cnn_pre)

figure()
plot(time(te_index), bagging_cnn_pre, 'b')
hold on
plot(time(te_index), te_value, 'r')
title('The performance of cnn')
legend('forecast', 'true')

%% aep dataset ----------------------------------------------------------
clear all
load ..\aep_dataset.mat
%% divide the training, validation, and testing dataset
start_index = 1+144*4*30+41 - 144*(30+30+30+30); 
te_index = [(1+144*4*30 + 41) : (1+144*4*30 + 41) + 7*144]; % one week to test
va_index = te_index - 7*144; % the previous week to validate
tr_index = [start_index : va_index(1)]; 
%% generate the training data_set and validation data_set and test data_set
% for cnn model input (input size equals 200)
% training data_set
% normalize 
[tr_dataset(:,1), tr1_min, tr1_gap] = minmaxnor(series_data(1:tr_index(end), 1));
[tr_dataset(:,2), tr2_min, tr2_gap] = minmaxnor(series_data(1:tr_index(end), 2));

k = 1;
for i = tr_index(1) : tr_index(end) - 200
    tr_x{k} = tr_dataset(i: i + 200-1, :)';
    tr_y{k} = tr_dataset(i + 200, 1)';
    k = k+1;
end
% construct the 4-D double  training dataset
for i = 1: size(tr_x, 2)
    tr_X(:, :, 1, i) = tr_x{i};
    tr_Y(i,:) = tr_y{i}; 
end

for i = 1:5
    % bagging the training dataset with 0.8
    r_index = randperm(size(tr_X,4));
    train_data = tr_X(:,:,:,r_index(1:floor(size(r_index, 2)*0.8)));
    train_y = tr_Y(r_index(1:floor(size(r_index, 2)*0.8)));
    
    % construct the CNN model
    layers = [
        imageInputLayer([2 200 1])
        convolution2dLayer([2 8],8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        averagePooling2dLayer([2,2])
        convolution2dLayer([1 4],16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'MiniBatchSize',300, ...
        'MaxEpochs',100, ...
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',20, ...
        'Plots','none', ...
        'ExecutionEnvironment', 'gpu',...
        'Verbose',false);
    
    [cnn_net{i}, info] = trainNetwork(tr_X,tr_Y,layers,options);
    training_loss{i} = info.TrainingRMSE;
end
save('case1_aep_bagging_cnn_model.mat', 'cnn_net')

load case1_aep_bagging_cnn_model.mat

% testing data_set
[te_dataset(te_index(1)-200: te_index(end), 1), te1_min, te1_gap] ...
            = minmaxnor(series_data(te_index(1)-200: te_index(end), 1));
[te_dataset(te_index(1)-200: te_index(end), 2), te2_min, te2_gap] ...
            = minmaxnor(series_data(te_index(1)-200: te_index(end), 2));

k = 1;
for i = te_index(1): te_index(end)
    te_x{k} = te_dataset(i-(200): i-1, :)';
    te_y{k} = te_dataset(i, 1)';
    k = k+1;
end
% construct the 4-D doubule 
for i = 1: size(te_x, 2)
    te_X(:, :, 1, i) = te_x{i};
    te_Y(i,:) = te_y{i}; 
end

% see the testing effect
for j = 1:5
    for i = 1: size(te_x, 2)
        YPre(j,i) = predict(cnn_net{j}, te_X(:, :, 1, i), 'ExecutionEnvironment','gpu');
    end
end
Ypre1 = mean(YPre, 1); % the bagging forecasting

% min max reverse
bagging_cnn_pre = versenor(Ypre1, te1_min, te1_gap);
save('case1_aep_bagging_cnn_forecast.mat', 'bagging_cnn_pre')

te_value = series_data(va_index, 1)';

% rmse
rmse_aep_bagging_cnn = rmse_1(te_value, bagging_cnn_pre)

figure()
plot(time(te_index), bagging_cnn_pre, 'b')
hold on
plot(time(te_index), te_value, 'r')
title('The performance of cnn')
legend('forecast', 'true')