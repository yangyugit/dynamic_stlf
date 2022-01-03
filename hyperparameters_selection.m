% hyperparameters selection 
% use LSTM model for AEP dataset as an example 
% NOTE: IT takes lots of time to run this code

% load the APE dataset (10-minutes-resolution)
load aep_dataset.mat

% divide the training, validation, and testing dataset
start_index = 1+144*4*30+41 - 144*(30+30+30+30); 
te_index = [(1+144*4*30 + 41) : (1+144*4*30 + 41) + 7*144];
va_index = te_index - 7*144; 
tr_index = [start_index : va_index(1)]; 

% define the training dataset
x_tr = series_data(tr_index, :); 
y_tr= series_data(tr_index +1, 1);
% max min normalize train data: nx_tr; ny_tr;
[nx_tr(:,1), nx1_min_tr, nx1_gap_tr] = minmaxnor(x_tr(:,1));
[nx_tr(:,2), nx2_min_tr, nx2_gap_tr] = minmaxnor(x_tr(:,2));
[ny_tr, ny_min_tr, ny_gap_tr] = minmaxnor(y_tr);

% validation dataset
x_va = series_data(va_index -1 , :); 
y_va = series_data(va_index, 1);
% max min normalize validation data: nx_ta; ny_ta;
[nx_va(:,1), nx1_min_va, nx1_gap_va] = minmaxnor(x_va(:,1));
[nx_va(:,2), nx2_min_va, nx2_gap_va] = minmaxnor(x_va(:,2));
[ny_va, ny_min_va, ny_gap_va] = minmaxnor(y_va);

% define the model
numFeatures = 2;
numResponses = 1;

layers{1} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(10)
    fullyConnectedLayer(numResponses)
    regressionLayer];

layers{2} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(20)
    fullyConnectedLayer(numResponses)
    regressionLayer];


layers{3} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50)
    fullyConnectedLayer(numResponses)
    regressionLayer];


layers{4} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(100)
    fullyConnectedLayer(numResponses)
    regressionLayer];


layers{5} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(200)
    fullyConnectedLayer(numResponses)
    regressionLayer];

layers{6} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(10)
    lstmLayer(10)
    fullyConnectedLayer(numResponses)
    regressionLayer];

layers{7} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(20)
    lstmLayer(20)
    fullyConnectedLayer(numResponses)
    regressionLayer];


layers{8} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50)
    lstmLayer(50)
    fullyConnectedLayer(numResponses)
    regressionLayer];


layers{9} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(100)
    lstmLayer(100)
    fullyConnectedLayer(numResponses)
    regressionLayer];


layers{10} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(200)
    lstmLayer(200)
    fullyConnectedLayer(numResponses)
    regressionLayer];

layers{11} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50)
    lstmLayer(50)
    lstmLayer(50)
    fullyConnectedLayer(numResponses)
    regressionLayer];

layers{12} = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(100)
    lstmLayer(100)
    lstmLayer(100)
    fullyConnectedLayer(numResponses)
    regressionLayer];



% the hyper-parameters
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'GradientThreshold',0.075, ...
    'InitialLearnRate',0.0025, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','none',...
    'ExecutionEnvironment', 'gpu'); % assign the cpu

% train the net
for i = 1:12
    [lstm_net{i}, info] = trainNetwork(nx_tr', ny_tr', layers{i}, options);
    training_loss{i} = info.TrainingRMSE;
end


% predict
for j = 1:12
    for i = 1: size(nx_va,1)
        [lstm_net{j}, lstm_prediction] = ...
            predictAndUpdateState(lstm_net{j}, nx_va(i, :)','ExecutionEnvironment','cpu');
        pre_lstm(i) = lstm_prediction;
    end
    y_lstm{j} = pre_lstm;
end

true_value = series_data(va_index, 1);
% mse 
for i = 1:12
    % min max reverse
    y_lstm{i} = versenor(y_lstm{i}, ny_min_va, ny_gap_va);
    MSE{i} = mse_1(y_lstm{i}, true_value');
end

MSE % choose the LSTM structure according to the MSE