% forecasting_by_single_method
%% ihepc dataset-----------------------------------------------------------
%-------------------------------------------------------------------------%
clear all

load ..\ihepc_dataset.mat
start_index = 37+360 + 15*24*60;  % time(start_index) is 01-Jan-2007 00:00:00
end_index = start_index + (31+28)*24*60; 

tr_index = [start_index : end_index]; 
va_index = [end_index: end_index + 7*24*60];
te_index = [end_index + 7*24*60 : end_index + 7*24*60+7*24*60];

time = time(1: te_index(end)+1); active_power = active_power(1:te_index(end)+1);

% record the true value of testing data
true_value = active_power(te_index)';
% add the measurement noise
[nor_data, min_data, gap_data] = minmaxnor(active_power);
measure_noise_var = 0.025;
measure_noise = normrnd(0, measure_noise_var, [1, size(active_power, 1)])';
nor_data = nor_data + measure_noise;
active_power = versenor(nor_data, min_data, gap_data);

%% lstm model
% testing dataset
x_te = active_power(te_index-1); y_te = active_power(te_index);
[nx_te, nx_min_te, nx_gap_te] = minmaxnor(x_te);
[ny_te, ny_min_te, ny_gap_te] = minmaxnor(y_te);

load ../test_case_1/case1_ihepc_lstm_model.mat
% see the validation effect
for i = 1: size(nx_te, 1)
    [lstm_net, lstm_prediction] = ...
        predictAndUpdateState(lstm_net, nx_te(i,:)','ExecutionEnvironment','gpu');
    y_lstm(i) = lstm_prediction;
end

% min max reverse
y_lstm = versenor(y_lstm, ny_min_te, ny_gap_te);
rmse_ihepc_lstm = rmse_1(true_value, y_lstm)

figure()
plot(time(te_index), y_lstm, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of lstm (IHEPC)')
legend('forecast', 'true')
save('case2_ihepc_lstm_forecast.mat','y_lstm');

% %% gpr model
% window_size = 5;
% for i = te_index(1):1:te_index(end)
%     gprMdl =  fitrgp(active_power(i-2 - window_size: i-2), active_power(i-1 - window_size: i-1));
%     y_gpr1(i-window_size: i, 1) = predict(gprMdl, active_power(i-1 - window_size:i-1));
%     y_gpr2(i) = y_gpr1(i, 1);
% end
% y_gpr = y_gpr2(te_index);
% 
% rmse_gpr_ihepc = rmse_1(true_value, y_gpr)
% 
% figure()
% plot(time(te_index), y_gpr,'b')
% hold on
% plot(time(te_index), true_value,'r')
% title('The performance of GPR')
% legend('forecast data', 'true data')
% save('case2_ihepc_gpr_forecast.mat','y_gpr');
% 
% %% var model
% Mdl = varm(1, 20);
% for i = te_index(1):1:te_index(end)
%     EstMdl = estimate(Mdl, active_power(i-400 :(i-1)));
%     y_var1(i) = forecast(EstMdl,1,active_power(i-400: (i-1)));
% end
% y_var = y_var1(te_index);
% 
% rmse_var_ihepc = rmse_1(true_value, y_var)
% 
% figure()
% plot(time(te_index), y_var,'b')
% hold on
% plot(time(te_index), true_value,'r')
% title('The performance of VAR')
% legend('forecast data', 'true data')
% save('case2_ihepc_var_forecast.mat','y_var');

%% aep dataset-------------------------------------------------------------
%-------------------------------------------------------------------------%
clear all
load ../aep_dataset.mat
start_index = 1+144*4*30+41 - 144*(30+30+30+30); 
te_index = [(1+144*4*30 + 41) : (1+144*4*30 + 41) + 7*144]; % one week to test
va_index = te_index - 7*144; % the previous week to validate
tr_index = [start_index : va_index(1)]; 

% record the true testing dataset
true_value = series_data(te_index, 1)';
% add the measurement noise
[nor1_data, min1_data, gap1_data] = minmaxnor(series_data(:, 1));
[nor2_data, min2_data, gap2_data] = minmaxnor(series_data(:, 2));
measure_noise_var = 0.02; 
measure_noise = normrnd(0, measure_noise_var, [1, size(series_data, 1)])';
nor_data1 = nor1_data + measure_noise;
nor_data2 = nor2_data + measure_noise;
series_data(:, 1) = versenor(nor_data1, min1_data, gap1_data);
series_data(:, 2) = versenor(nor_data2, min2_data, gap2_data);
%% lstm model
% testing dataset
x_te = series_data(te_index -1 , :); y_te= series_data(te_index, 1);
% max min normalize test data: nx_te; ny_te;
[nx_te(:,1), nx1_min_te, nx1_gap_te] = minmaxnor(x_te(:,1));
[nx_te(:,2), nx2_min_te, nx2_gap_te] = minmaxnor(x_te(:,2));
[ny_te, ny_min_te, ny_gap_te] = minmaxnor(y_te);

load ../test_case_1/case1_aep_lstm_model.mat
for i = 1: size(nx_te,1)
    [lstm_net, lstm_prediction] = ...
        predictAndUpdateState(lstm_net, nx_te(i,:)','ExecutionEnvironment','gpu');
    y_lstm(i) = lstm_prediction;
end

% min max reverse
y_lstm = versenor(y_lstm, ny_min_te, ny_gap_te);
rmse_aep_lstm = rmse_1(true_value, y_lstm)

figure()
plot(time(te_index), y_lstm, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of lstm (IHEPC)')
legend('forecast', 'true')
save('case2_aep_lstm_forecast.mat','y_lstm');

% %% gpr model
% window_size = 5;
% for i = te_index(1):1:te_index(end)
%     gprMdl =  fitrgp(series_data(i-2 - window_size: i-2, :), series_data(i-1 - window_size: i-1, 1));
%     y_gpr1(i-window_size: i, 1) = predict(gprMdl, series_data(i-1 - window_size:i-1, :));
%     y_gpr2(i) = y_gpr1(i, 1);
% end
% y_gpr = y_gpr2(te_index);
% rmse_aep_gpr = rmse_1(true_value, y_gpr)
% save('case2_aep_gpr_forecast.mat','y_gpr');
% 
% figure()
% plot(time(te_index), y_gpr,'b')
% hold on
% plot(time(te_index), true_value,'r')
% title('The performance of GPR')
% legend('forecast data', 'true data')
% 
% %% var model
% Mdl = varm(2, 10);
% for i = te_index(1):1:te_index(end)
%     EstMdl = estimate(Mdl,series_data(i-200 :(i-1),:));
%     y_var1(i,:) = forecast(EstMdl,1,series_data(i-200 :(i-1),:));
% end
% y_var = y_var1(te_index);
% rmse_aep_var = rmse_1(true_value, y_var)
% save('case2_aep_var_forecast.mat','y_var')
% figure()
% plot(time(te_index), y_var,'b')
% hold on
% plot(time(te_index), true_value,'r')
% title('The performance of VAR')
% legend('forecast data', 'true data')