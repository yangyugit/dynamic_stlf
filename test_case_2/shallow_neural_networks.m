%  ELM FNN and RBF with the trim ensemble method
%% IHEPC dataset===========================================================
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
x_tr = active_power(tr_index)';  y_tr= active_power(tr_index +1)';
x_va = active_power(va_index-1)'; y_va = active_power(va_index)';
va_value = active_power(va_index)';

% add the measurement noise
[nor_data, min_data, gap_data] = minmaxnor(active_power);
measure_noise_var = 0.025;
measure_noise = normrnd(0, measure_noise_var, [1, size(active_power, 1)])';
nor_data = nor_data + measure_noise;
active_power = versenor(nor_data, min_data, gap_data);

x_te = active_power(te_index-1)'; y_te = active_power(te_index)';
%% fnn_net
fnn_net = feedforwardnet(100);
fnn_net.trainParam.showWindow = false;
fnn_net.trainParam.showCommandLine = false;
% training
fnn_net = train(fnn_net, x_tr, y_tr);
% validation performance
for i = 1:size(x_va, 2)
    fnn_va_pre(i) = fnn_net(x_va(i));
end
rmse_ihepc_fnn_va = rmse_1(va_value, fnn_va_pre)
figure()
plot(time(va_index), fnn_va_pre, 'b')
hold on
plot(time(va_index), va_value, 'r')
title('The performance of fnn\_net in validation dataset (IHEPC)')
legend('forecast', 'true')
% testing performance
for i = 1:size(x_te, 2)
    fnn_te_pre(i) = fnn_net(x_te(i));
end
rmse_ihepc_fnn_te = rmse_1(true_value, fnn_te_pre)
figure()
plot(time(te_index), fnn_te_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of fnn\_net in testing dataset (IHEPC)')
legend('forecast', 'true')

%% elm_net (unstable performance)
% max min normalize data
[nx_tr, nx_min_tr, nx_gap_tr] = minmaxnor(x_tr(end- size(x_te, 2):end));
[ny_tr, ny_min_tr, ny_gap_tr] = minmaxnor(y_tr(end- size(y_te, 2):end));
[nx_te, nx_min_te, nx_gap_te] = minmaxnor(x_te);
[ny_te, ny_min_te, ny_gap_te] = minmaxnor(y_te);
[nx_va, nx_min_va, nx_gap_va] = minmaxnor(x_va);
[ny_va, ny_min_va, ny_gap_va] = minmaxnor(y_va);
% training
for i = 1:size(nx_tr, 2)
    tr_x{i} = nx_tr(i);
    tr_y{i} = ny_tr(i);
end
elm_net = elmannet(1:2, 100);
elm_net.trainParam.showWindow = false;
elm_net.trainParam.showCommandLine = false; 
[Xs,Xi,Ai,Ts] = preparets(elm_net, tr_x, tr_y);
elm_net = train(elm_net,Xs,Ts,Xi,Ai);
% validation performance
for i = 1:size(nx_va, 2)
    elm_va_pre(i) = sim(elm_net, nx_va(i));
end
elm_va_pre = versenor(elm_va_pre, ny_min_va, ny_gap_va);
rmse_ihepc_elm_va = rmse_1(va_value, elm_va_pre)
figure()
plot(time(va_index), elm_va_pre, 'b')
hold on
plot(time(va_index), va_value, 'r')
title('The performance of elm\_net in validation dataset (IHEPC)')
legend('forecast', 'true')
% testing performance
for i = 1:size(nx_te, 2)
    elm_te_pre(i) = sim(elm_net, nx_te(i));
end
elm_te_pre = versenor(elm_te_pre, ny_min_te, ny_gap_te);
rmse_ihepc_elm_te = rmse_1(true_value, elm_te_pre)
figure()
plot(time(te_index), elm_te_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of elm\_net in testing dataset (IHEPC)')
legend('forecast', 'true')
%% rbf_net (the computation is too complex)
%training
rbf_net = newrb(x_tr(end- floor(0.15*size(x_va, 2)): end), y_tr(end- floor(0.15*size(y_va, 2)): end));

% validation performance
for i = 1:size(x_va, 2)
    rbf_va_pre(i) = sim(rbf_net, x_va(i));
end
rmse_ihepc_rbf_va = rmse_1(va_value, rbf_va_pre)
figure()
plot(time(va_index), rbf_va_pre, 'b')
hold on
plot(time(va_index), va_value, 'r')
title('The performance of rbf\_net in validation dataset (IHEPC)')
legend('forecast', 'true')
% testing performance

for i = 1:size(x_te, 2)
    rbf_te_pre(i) = sim(rbf_net, x_te(i));
end

rmse_ihepc_rbf_te = rmse_1(true_value, rbf_te_pre)
figure()
plot(time(te_index), rbf_te_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of rbf\_net in testing dataset (IHEPC)')
legend('forecast', 'true')

%% ensemble forecasting based on trim_aggregation
trim_para = trim_par_cal(rmse_ihepc_fnn_va, rmse_ihepc_elm_va, rmse_ihepc_rbf_va);
ensemble_fore = trim_agg(trim_para, fnn_te_pre, elm_te_pre, rbf_te_pre);
ihepc_rmse = rmse_1(true_value, ensemble_fore) 

figure()
plot(time(te_index), ensemble_fore, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of ann\_trim\_agg in testing dataset (IHEPC)')
legend('forecast', 'true')

save('case2_ihepc_trim_agg.mat','ensemble_fore')



%% AEP dataset=============================================================
clear all
load ..\aep_dataset.mat
%% divide the training, validation, and testing dataset
start_index = 1+144*4*30+41 - 144*(30+30+30+30); 
te_index = [(1+144*4*30 + 41) : (1+144*4*30 + 41) + 7*144]; % one week to test
va_index = te_index - 7*144; % the previous week to validate
tr_index = [start_index : va_index(1)]; 

% record the true testing dataset
true_value = series_data(te_index, 1)';
va_value = series_data(va_index, 1)';
x_tr = series_data(tr_index, 1)';  y_tr= series_data(tr_index +1, 1)';
x_va = series_data(va_index-1, 1)';  y_va= series_data(va_index, 1)';

% add the measurement noise
[nor_data, min_data, gap_data] = minmaxnor(series_data(:, 1));
measure_noise_var = 0.025; 
measure_noise = normrnd(0, measure_noise_var, [1, size(series_data, 1)])';
nor_data = nor_data + measure_noise;
series_data(:, 1) = versenor(nor_data, min_data, gap_data);

x_te = series_data(te_index-1, 1)'; y_te = series_data(te_index, 1)';

%% fnn_net
% training 
fnn_net = feedforwardnet(100);
fnn_net.trainParam.showWindow = false;
fnn_net.trainParam.showCommandLine = false;
fnn_net = train(fnn_net, x_tr, y_tr);
% validation performance
for i = 1:size(x_va, 2)
    fnn_va_pre(i) = fnn_net(x_va(i));
end
rmse_aep_fnn_va = rmse_1(va_value, fnn_va_pre)
figure()
plot(time(va_index), fnn_va_pre, 'b')
hold on
plot(time(va_index), va_value, 'r')
title('The performance of fnn\_net in validation dataset (AEP)')
legend('forecast', 'true')
% testing performance
for i = 1:size(x_te, 2)
    fnn_te_pre(i) = fnn_net(x_te(i));
end

rmse_aep_fnn_te = rmse_1(true_value, fnn_te_pre)
figure()
plot(time(te_index), fnn_te_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of fnn in testing dataset (AEP)')
legend('forecast', 'true')

%% elm_net (ihepc is not well, also in aep, unstable)
% max min normalize train data
[nx_tr, nx_min_tr, nx_gap_tr] = minmaxnor(x_tr(end- size(x_te, 2):end));
[ny_tr, ny_min_tr, ny_gap_tr] = minmaxnor(y_tr(end- size(y_te, 2):end));
[nx_va, nx_min_va, nx_gap_va] = minmaxnor(x_va);
[ny_va, ny_min_va, ny_gap_va] = minmaxnor(y_va);
[nx_te, nx_min_te, nx_gap_te] = minmaxnor(x_te);
[ny_te, ny_min_te, ny_gap_te] = minmaxnor(y_te);
% training
for i = 1:size(nx_tr, 2)
    tr_x{i} = nx_tr(i);
    tr_y{i} = ny_tr(i);
end
elm_net = elmannet(1:2, 100);
elm_net.trainParam.showWindow = false;
elm_net.trainParam.showCommandLine = false; 
[Xs,Xi,Ai,Ts] = preparets(elm_net, tr_x, tr_y);
elm_net = train(elm_net,Xs,Ts,Xi,Ai);

% validation performance
for i = 1:size(nx_va, 2)
    elm_va_pre(i) = sim(elm_net, nx_va(i));
end

elm_va_pre = versenor(elm_va_pre, ny_min_va, ny_gap_va);
rmse_aep_elm_va = rmse_1(va_value, elm_va_pre)

figure()
plot(time(va_index), elm_va_pre, 'b')
hold on
plot(time(va_index), va_value, 'r')
title('The performance of elm\_net in validation dataset (AEP)')
legend('forecast', 'true')

% testing performance
for i = 1:size(nx_te, 2)
    elm_te_pre(i) = sim(elm_net, nx_te(i));
end

elm_te_pre = versenor(elm_te_pre, ny_min_te, ny_gap_te);
rmse_aep_elm_te = rmse_1(elm_te_pre, true_value)

figure()
plot(time(te_index), elm_te_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of elm\_net in testing dataset (AEP)')
legend('forecast', 'true')

%% rbf_net
% training
rbf_net = newrb(x_tr(end- size(x_te, 2):end), y_tr(end- size(y_te, 2):end));

% validation performance
for i = 1:size(x_va, 2)
    rbf_va_pre(i) = sim(rbf_net, x_va(i));
end
rmse_aep_rbf_va = rmse_1(va_value, rbf_va_pre)
figure()
plot(time(va_index), rbf_va_pre, 'b')
hold on
plot(time(va_index), va_value, 'r')
title('The performance of rbf\_net in validation dataset (AEP)')
legend('forecast', 'true')

% testing performance
for i = 1:size(x_te, 2)
    rbf_te_pre(i) = sim(rbf_net, x_te(i));
end
rmse_aep_rbf_te = rmse_1(true_value, rbf_te_pre)
figure()
plot(time(te_index), rbf_te_pre, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of rbf\_net in testing dataset (AEP)')
legend('forecast', 'true')

%% ensemble forecasting based on trim_aggregation
trim_para = trim_par_cal(rmse_aep_fnn_va, rmse_aep_elm_va, rmse_aep_rbf_va);
ensemble_fore = trim_agg(trim_para, fnn_te_pre, elm_te_pre, rbf_te_pre);

ape_rmse = rmse_1(true_value, ensemble_fore)

figure()
plot(time(te_index), ensemble_fore, 'b')
hold on
plot(time(te_index), true_value, 'r')
title('The performance of ann\_trim\_agg in testing dataset (AEP)')
legend('forecast', 'true')

save('case2_aep_trim_agg.mat','ensemble_fore')