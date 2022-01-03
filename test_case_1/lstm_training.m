% training deep learning model i.e., LSTM and CNN

%% IHEPC dataset (1-minute resolution)======================================
clear all
load ..\ihepc_dataset.mat
%% divide the training, validation, and testing dataset
start_index = 37+360 + 15*24*60;  % time(start_index) is 01-Jan-2007 00:00:00
end_index = start_index + (31+28)*24*60; 

tr_index = [start_index : end_index]; 
va_index = [end_index: end_index + 7*24*60];
te_index = [end_index + 7*24*60 : end_index + 7*24*60+7*24*60];

time = time(1: te_index(end)+1); active_power = active_power(1:te_index(end)+1);

% training data
x_tr = active_power(tr_index);  y_tr= active_power(tr_index +1);
% max min normalize train data: nx_tr; ny_tr;
[nx_tr, nx_min_tr, nx_gap_tr] = minmaxnor(x_tr);
[ny_tr, ny_min_tr, ny_gap_tr] = minmaxnor(y_tr);

% validation dataset
x_va = active_power(va_index-1); y_va = active_power(va_index);
[nx_va, nx_min_va, nx_gap_va] = minmaxnor(x_va);
[ny_va, ny_min_va, ny_gap_va] = minmaxnor(y_va);

%% lstm model training
% define the model
numFeatures = 1;
numResponses = 1;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(200)
    lstmLayer(200)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% the hyper-parameters
options = trainingOptions('adam', ...
    'MiniBatchSize', 300,...
    'MaxEpochs',30, ...
    'GradientThreshold',0.075, ...
    'InitialLearnRate',0.0025, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',15, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'gpu'); % assign the gpu

% train the net
[lstm_net, info] = trainNetwork(nx_tr', ny_tr', layers, options);
training_loss = info.TrainingRMSE;
save('case1_ihepc_lstm_model.mat', 'lstm_net')

load case1_ihepc_lstm_model.mat
% see the validation effect
for i = 1: size(nx_va,1)
[lstm_net, lstm_prediction] = ...
    predictAndUpdateState(lstm_net, nx_va(i)','ExecutionEnvironment','gpu');
y_lstm(i) = lstm_prediction;
end

% min max reverse
y_lstm = versenor(y_lstm, ny_min_va, ny_gap_va);
true_value = active_power(va_index)';
rmse_lstm_ihepc = rmse_1(true_value, y_lstm)

figure()
plot(time(va_index), y_lstm, 'b')
hold on
plot(time(va_index), true_value, 'r')
title('The performance of lstm (IHEPC)')
legend('forecast', 'true')

%% AEP dataset (10-minutes resolution)=====================================
clear all
load ..\aep_dataset.mat
%% divide the training, validation, and testing dataset
start_index = 1+144*4*30+41 - 144*(30+30+30+30); 
te_index = [(1+144*4*30 + 41) : (1+144*4*30 + 41) + 7*144]; % one week to test
va_index = te_index - 7*144; % the previous week to validate
tr_index = [start_index : va_index(1)]; 

% training dataset
x_tr = series_data(tr_index, :); 
y_tr= series_data(tr_index +1, 1);
% max min normalize train data: nx_tr; ny_tr;
[nx_tr(:,1), nx1_min_tr, nx1_gap_tr] = minmaxnor(x_tr(:,1));
[nx_tr(:,2), nx2_min_tr, nx2_gap_tr] = minmaxnor(x_tr(:,2));
[ny_tr, ny_min_tr, ny_gap_tr] = minmaxnor(y_tr);

% valadation dataset
x_va = series_data(va_index -1 , :);  y_va = series_data(va_index, 1);
% max min normalize test data: nx_te; ny_te;
[nx_va(:,1), nx1_min_va, nx1_gap_va] = minmaxnor(x_va(:,1));
[nx_va(:,2), nx2_min_va, nx2_gap_va] = minmaxnor(x_va(:,2));
[ny_va, ny_min_va, ny_gap_va] = minmaxnor(y_va);

%% lstm model training
% define the model
numFeatures = 2;
numResponses = 1;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(100)
    lstmLayer(100)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% the hyper-parameters
options = trainingOptions('adam', ...
    'MiniBatchSize', 128,...
    'MaxEpochs',30, ...
    'GradientThreshold',0.075, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'gpu'); % assign the cpu

% train the net
[lstm_net, info] = trainNetwork(nx_tr', ny_tr', layers, options);
training_loss = info.TrainingRMSE;
save('case1_aep_lstm_model.mat', 'lstm_net')

load case1_aep_lstm_model.mat
% see the validation effect
for i = 1: size(nx_va,1)
    [lstm_net, lstm_prediction] = ...
        predictAndUpdateState(lstm_net, nx_va(i,:)','ExecutionEnvironment','gpu');
    y_lstm(i) = lstm_prediction;
end

% min max reverse
y_lstm = versenor(y_lstm, ny_min_va, ny_gap_va);
true_value = series_data(va_index, 1)';
rmse_lstm_aep = rmse_1(true_value, y_lstm)
figure()
title('performance of lstm')
plot(time(va_index), y_lstm, 'b')
hold on
plot(time(va_index), true_value, 'r')
legend('forecast', 'true')