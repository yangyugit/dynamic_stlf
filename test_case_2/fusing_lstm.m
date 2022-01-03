% fusing LSTM
%% IHEPC dataset (1-minute resolution)
clear all
load ..\ihepc_dataset.mat
%% divide the training, validation, and testing dataset
start_index = 37+360 + 15*24*60;  % time(start_index) is 01-Jan-2007 00:00:00
end_index = start_index + (31+28)*24*60; 
tr_index = [start_index : end_index]; 
va_index = [end_index: end_index + 7*24*60];
te_index = [end_index + 7*24*60 : end_index + 7*24*60+7*24*60];
time = time(1: te_index(end)+1); active_power = active_power(1:te_index(end)+1);
% record the true value of testing dataset
true_value = active_power(te_index)';

% add the measurement noise
[nor_data, min_data, gap_data] = minmaxnor(active_power);
measure_noise_var = 0.025;
measure_noise = normrnd(0, measure_noise_var, [1, size(active_power, 1)])';
nor_data = nor_data + measure_noise;
active_power = versenor(nor_data, min_data, gap_data);

load ..\test_case_1\case1_ihepc_fusing_lstm_model.mat
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
rmse_ihepc_fusing = rmse_1(true_value, y_fusing)
save('case2_ihepc_fusing_lstm_forecast.mat', 'fusing_lstm_pre')
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
% save the true_value of the testing dataset
true_value = series_data(te_index, 1)';
% add the measurement noise (only the power meter noise)
[nor_data, min_data, gap_data] = minmaxnor(series_data(:, 1));
measure_noise_var = 0.025; 
measure_noise = normrnd(0, measure_noise_var, [1, size(series_data, 1)])';
nor_data = nor_data + measure_noise;
series_data(:, 1) = versenor(nor_data, min_data, gap_data);

load ..\test_case_1\case1_aep_fusing_lstm_model.mat
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
rmse_aep_fusing = rmse_1(true_value, y_lstm)
save('case2_aep_fusing_lstm_forecast','fusing_lstm_pre')
figure()
plot(time(te_index), fusing_lstm_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of lstm (AEP)')
legend('forecast', 'true')