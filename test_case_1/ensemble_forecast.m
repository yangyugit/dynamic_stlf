% dynamic ensemble forecasting combining var, gpr, and lstm

%% ihepc dataset
clear all
load ..\ihepc_dataset.mat
start_index = 37+360 + 15*24*60;  % time(start_index) is 01-Jan-2007 00:00:00
end_index = start_index + (31+28)*24*60; 

tr_index = [start_index : end_index]; 
va_index = [end_index: end_index + 7*24*60];
te_index = [end_index + 7*24*60 : end_index + 7*24*60+7*24*60];

time = time(1: te_index(end)+1); active_power = active_power(1:te_index(end)+1);

% forecast by single model i.e., var, gpr, and lstm
load case1_ihepc_var_forecast.mat; load case1_ihepc_gpr_forecast.mat; 
load case1_ihepc_lstm_forecast.mat;
% testing data
te_value = active_power(te_index)';

% static estimation for the weighting vector 
window_size = 50; member_model = [y_var; y_gpr; y_lstm]; 
w_s = static_est(member_model, te_value, window_size);

t_static(1) = te_value(1);
% test the weighting vector used for model combining
for i = 2:size(time(te_index), 1)
    t_static(i) = w_s(:, i-1)' * member_model(: ,i);
end

% Dynamic estimation
r_svar = 1e-6; s_svar= 1e-6;
w_d = pf_dynamic_est(w_s, r_svar, s_svar);
t_dynamic(1) = te_value(1);
for i = 2:size(time(te_index), 1)
    t_dynamic(i) = w_d(:, i-1)' * member_model(: ,i);
end

% error indices
[var_rmse, var_mae, var_mape] = error_indices(y_var, te_value);
v_index = [var_rmse, var_mae, var_mape]';

[gpr_rmse, gpr_mae, gpr_mape] = error_indices(y_gpr, te_value);
g_index = [gpr_rmse, gpr_mae, gpr_mape]';

[lstm_rmse, lstm_mae, lstm_mape] = error_indices(y_lstm, te_value);
l_index = [lstm_rmse, lstm_mae, lstm_mape]';

[s_rmse, s_mae, s_mape] = error_indices(t_static, te_value); 
s_index = [s_rmse, s_mae, s_mape]';

[d_rmse, d_mae, d_mape] = error_indices(t_dynamic, te_value); 
d_index  = [d_rmse, d_mae, d_mape]';

load case1_ihepc_bagging_cnn_forecast; load case1_ihepc_fusing_lstm_forecast;
load case1_ihepc_trim_agg.mat;

[bagging_cnn_rmse, bagging_cnn_mae, bagging_cnn_mape] = error_indices(bagging_cnn_pre, te_value); 
bagging_cnn_index  = [bagging_cnn_rmse, bagging_cnn_mae, bagging_cnn_mape]';

[fusing_lstm_rmse, fusing_lstm_mae, fusing_lstm_mape] = error_indices(fusing_lstm_pre, te_value); 
fusing_lstm_index  = [fusing_lstm_rmse, fusing_lstm_mae, fusing_lstm_mape]';

[trim_agg_rmse, trim_agg_mae, trim_agg_mape] = error_indices(ensemble_fore, te_value); 
trim_agg_index  = [trim_agg_rmse, trim_agg_mae, trim_agg_mape]';
% csv
datacolumns = {'var','gpr','lstm','static', 'dynamic', 'trim_agg' ,'bagging cnn', 'fusing lstm'};
data = table(v_index, g_index, l_index, s_index, d_index, trim_agg_index, bagging_cnn_index, fusing_lstm_index, 'VariableNames', datacolumns);
writetable(data, 'case1_ihepc_error_indices.csv')

save test_case1_ihepc.mat

% plot
figure()
title('The performance of dynamic ensemble method')
plot(time(te_index), te_value, '-', 'LineWidth', 1, 'color', [255, 0, 0]./255)
hold on 
plot(time(te_index), t_static, 'o-', 'LineWidth', 1,  'color', [0, 255, 0]./255)
plot(time(te_index), t_dynamic, 's-', 'LineWidth', 1, 'color', [0, 0, 255]./255)
plot(time(te_index), ensemble_fore, '^-', 'LineWidth', 1, 'color', [255, 0, 255]./255)
plot(time(te_index), bagging_cnn_pre, '<-', 'LineWidth', 1, 'color', [255, 255, 0]./255)
plot(time(te_index), fusing_lstm_pre, '>-', 'LineWidth', 1, 'color', [0, 255, 255]./255)

xlabel('time')
ylabel('residential load')
set(gca,'FontSize',16);
legend('true value','static ensemble method', 'dynamic ensemble method', 'trim_aggregation', 'bagging cnn', 'fusing lstm')


%% aep dataset
clear all
load ..\aep_dataset.mat
start_index = 1+144*4*30+41 - 144*(30+30+30+30); 
te_index = [(1+144*4*30 + 41) : (1+144*4*30 + 41) + 7*144]; % one week to test
va_index = te_index - 7*144; % the previous week to validate
tr_index = [start_index : va_index(1)]; 

% forecast by single model i.e., var, gpr, and lstm
load case1_aep_var_forecast.mat; load case1_aep_gpr_forecast.mat; 
load case1_aep_lstm_forecast.mat;
% testing data
te_value = series_data(te_index, 1)';

% static estimation for the weighting vector 
window_size = 50; member_model = [y_var; y_gpr; y_lstm]; 
w_s = static_est(member_model, te_value, window_size);

t_static(1) = te_value(1);
% test the weighting vector used for model combining
for i = 2:size(time(te_index), 1)
    t_static(i) = w_s(:, i-1)' * member_model(: ,i);
end

% Dynamic estimation
r_svar = 1e-6; s_svar= 1e-6;
w_d = pf_dynamic_est(w_s, r_svar, s_svar);
t_dynamic(1) = te_value(1);
for i = 2:size(time(te_index), 1)
    t_dynamic(i) = w_d(:, i-1)' * member_model(: ,i);
end

% error indices
[var_rmse, var_mae, var_mape] = error_indices(y_var, te_value);
v_index = [var_rmse, var_mae, var_mape]';

[gpr_rmse, gpr_mae, gpr_mape] = error_indices(y_gpr, te_value);
g_index = [gpr_rmse, gpr_mae, gpr_mape]';

[lstm_rmse, lstm_mae, lstm_mape] = error_indices(y_lstm, te_value);
l_index = [lstm_rmse, lstm_mae, lstm_mape]';

[s_rmse, s_mae, s_mape] = error_indices(t_static, te_value); 
s_index = [s_rmse, s_mae, s_mape]';

[d_rmse, d_mae, d_mape] = error_indices(t_dynamic, te_value); 
d_index  = [d_rmse, d_mae, d_mape]';

load case1_aep_bagging_cnn_forecast; load case1_aep_fusing_lstm_forecast;
load case1_aep_trim_agg.mat;

[bagging_cnn_rmse, bagging_cnn_mae, bagging_cnn_mape] = error_indices(bagging_cnn_pre, te_value); 
bagging_cnn_index  = [bagging_cnn_rmse, bagging_cnn_mae, bagging_cnn_mape]';

[fusing_lstm_rmse, fusing_lstm_mae, fusing_lstm_mape] = error_indices(fusing_lstm_pre, te_value); 
fusing_lstm_index  = [fusing_lstm_rmse, fusing_lstm_mae, fusing_lstm_mape]';

[trim_agg_rmse, trim_agg_mae, trim_agg_mape] = error_indices(ensemble_fore, te_value); 
trim_agg_index  = [trim_agg_rmse, trim_agg_mae, trim_agg_mape]';
% csv
datacolumns = {'var','gpr','lstm','static', 'dynamic', 'trim_agg' ,'bagging cnn', 'fusing lstm'};
data = table(v_index, g_index, l_index, s_index, d_index, trim_agg_index, bagging_cnn_index, fusing_lstm_index, 'VariableNames', datacolumns);
writetable(data, 'case1_aep_error_indices.csv')

save test_case1_aep.mat

% plot
figure()
title('The performance of dynamic ensemble method')
plot(time(te_index), te_value, '-', 'LineWidth', 1, 'color', [255, 0, 0]./255)
hold on 
plot(time(te_index), t_static, 'o-', 'LineWidth', 1,  'color', [0, 255, 0]./255)
plot(time(te_index), t_dynamic, 's-', 'LineWidth', 1, 'color', [0, 0, 255]./255)
plot(time(te_index), ensemble_fore, '^-', 'LineWidth', 1, 'color', [255, 0, 255]./255)
plot(time(te_index), bagging_cnn_pre, '<-', 'LineWidth', 1, 'color', [255, 255, 0]./255)
plot(time(te_index), fusing_lstm_pre, '>-', 'LineWidth', 1, 'color', [0, 255, 255]./255)

xlabel('time')
ylabel('residential load')
set(gca,'FontSize',16);
legend('true value','static ensemble method', 'dynamic ensemble method', 'trim_aggregation', 'bagging cnn', 'fusing lstm')
