% fusing LSTM
% % IHEPC dataset (1-minute resolution)
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

input = sequenceInputLayer(numFeatures,'Name','input');
fullyconnect = fullyConnectedLayer(numResponses,'Name','fully');
output = regressionLayer('Name','output');

concat = additionLayer(2,'Name','concat_1');

lstm_1 = lstmLayer(200,'Name','lstm_1');
lstm_2 = lstmLayer(200,'Name','lstm_2');

lgraph = layerGraph;
lgraph = addLayers(lgraph,input);
lgraph = addLayers(lgraph,lstm_1);
lgraph = addLayers(lgraph,lstm_2);
lgraph = addLayers(lgraph,concat);
lgraph = addLayers(lgraph,fullyconnect);
lgraph = addLayers(lgraph,output);

lgraph = connectLayers(lgraph, 'input', 'lstm_1');
lgraph = connectLayers(lgraph, 'input', 'lstm_2');
lgraph = connectLayers(lgraph,'lstm_1','concat_1/in1');
lgraph = connectLayers(lgraph,'lstm_2','concat_1/in2');
lgraph = connectLayers(lgraph, 'concat_1', 'fully');
lgraph = connectLayers(lgraph, 'fully', 'output');

% the hyper-parameters
options = trainingOptions('adam', ...
    'MiniBatchSize', 300,...
    'MaxEpochs',50, ...
    'GradientThreshold',0.075, ...
    'InitialLearnRate',0.0025, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'gpu'); % assign the gpu

% train the net
[fusing_net, info] = trainNetwork(nx_tr', ny_tr', lgraph, options);
training_loss = info.TrainingRMSE;
% 
save('case1_ihepc_fusing_lstm_model.mat', 'fusing_net')

load case1_ihepc_fusing_lstm_model.mat
% testing performance (testing data)
x_te = active_power(te_index-1); y_te = active_power(te_index);
[nx_te, nx_min_te, nx_gap_te] = minmaxnor(x_te);
[ny_te, ny_min_te, ny_gap_te] = minmaxnor(y_te);

for i = 1: size(nx_te, 1)
    [fusing_net, fusing_prediction] = ...
        predictAndUpdateState(fusing_net, nx_te(i,:)','ExecutionEnvironment','gpu');
    y_fusing(i) = fusing_prediction;
end

% min max reverse
fusing_lstm_pre = versenor(y_fusing, ny_min_te, ny_gap_te);
true_value = active_power(te_index)';
rmse_ihepc_fusing = rmse_1(true_value, y_fusing)
save('case1_ihepc_fusing_lstm_forecast.mat', 'fusing_lstm_pre')
figure()
plot(time(te_index), fusing_lstm_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of lstm (IHEPC)')
legend('forecast', 'true')
%% AEP dataset
clear all
load ../aep_dataset.mat
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

%% lstm model training
% define the model
numFeatures = 2;
numResponses = 1;

input = sequenceInputLayer(numFeatures,'Name','input');
fullyconnect = fullyConnectedLayer(numResponses,'Name','fully');
output = regressionLayer('Name','output');

concat = additionLayer(2,'Name','concat_1');

lstm_1 = lstmLayer(100,'Name','lstm_1');
lstm_2 = lstmLayer(100,'Name','lstm_2');

lgraph = layerGraph;
lgraph = addLayers(lgraph,input);
lgraph = addLayers(lgraph,lstm_1);
lgraph = addLayers(lgraph,lstm_2);
lgraph = addLayers(lgraph,concat);
lgraph = addLayers(lgraph,fullyconnect);
lgraph = addLayers(lgraph,output);

lgraph = connectLayers(lgraph, 'input', 'lstm_1');
lgraph = connectLayers(lgraph, 'input', 'lstm_2');
lgraph = connectLayers(lgraph,'lstm_1','concat_1/in1');
lgraph = connectLayers(lgraph,'lstm_2','concat_1/in2');
lgraph = connectLayers(lgraph, 'concat_1', 'fully');
lgraph = connectLayers(lgraph, 'fully', 'output');

% the hyper-parameters
options = trainingOptions('adam', ...
    'MiniBatchSize', 300,...
    'MaxEpochs',50, ...
    'GradientThreshold',0.075, ...
    'InitialLearnRate',0.0025, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'gpu'); % assign the gpu

% train the net
[fusing_net, info] = trainNetwork(nx_tr', ny_tr', lgraph, options);
training_loss = info.TrainingRMSE;
% 
save('case1_aep_fusing_lstm_model.mat', 'fusing_net')
load case1_aep_fusing_lstm_model.mat

% testing dataset
x_te = series_data(te_index -1 , :); y_te= series_data(te_index, 1);
% max min normalize test data: nx_te; ny_te;
[nx_te(:,1), nx1_min_te, nx1_gap_te] = minmaxnor(x_te(:,1));
[nx_te(:,2), nx2_min_te, nx2_gap_te] = minmaxnor(x_te(:,2));
[ny_te, ny_min_te, ny_gap_te] = minmaxnor(y_te);

for i = 1: size(nx_te,1)
    [fusing_net, lstm_prediction] = ...
        predictAndUpdateState(fusing_net, nx_te(i,:)','ExecutionEnvironment','gpu');
    y_lstm(i) = lstm_prediction;
end
% min max reverse
fusing_lstm_pre = versenor(y_lstm, ny_min_te, ny_gap_te);
true_value = series_data(te_index, 1)';
rmse_aep_fusing = rmse_1(true_value, y_lstm)
save('case1_aep_fusing_lstm_forecast','fusing_lstm_pre')
figure()
plot(time(te_index), fusing_lstm_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of lstm (IHEPC)')
legend('forecast', 'true')