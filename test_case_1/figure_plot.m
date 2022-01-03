% figure_plot
clear all
load test_case1_ihepc.mat

% figure()
% plot(time(te_index), te_value, '-', 'LineWidth', 1, 'color', [255, 0, 0]./255)
% hold on 
% plot(time(te_index), t_static, 'o-', 'LineWidth', 1,  'color', [0, 255, 0]./255)
% plot(time(te_index), t_dynamic, 's-', 'LineWidth', 1, 'color', [0, 0, 255]./255)
% % plot(time(te_index), ensemble_fore, '^-', 'LineWidth', 1, 'color', [255, 0, 255]./255)
% % plot(time(te_index), bagging_cnn_pre, '<-', 'LineWidth', 1, 'color', [255, 255, 0]./255)
% % plot(time(te_index), fusing_lstm_pre, '>-', 'LineWidth', 1, 'color', [0, 255, 255]./255)
% legend('true value', 'static ensemble', 'dynamic ensemble')
% set(gcf,'position',[10 10 600 300])

figure()
plot(time(te_index(1:60:end)), te_value(1:60:end), '-', 'LineWidth', 1.5, 'color', [255, 0, 0]./255)
hold on 
plot(time(te_index(1:60:end)), t_static(1:60:end), 'o-', 'LineWidth', 1.5,  'color', [0, 255, 0]./255)
plot(time(te_index(1:60:end)), t_dynamic(1:60:end), 's-', 'LineWidth', 1.5, 'color', [0, 0, 255]./255)
xlabel('time')
ylabel('active power load')
legend('true value', 'static ensemble', 'dynamic ensemble', 'Fontsize', 15)
set(gca,'FontSize',20);
set(gcf,'position',[5 5 1600 600])
print(gcf,'test11a','-dpng','-r600')

% plot(time(te_index(1:60:end)), ensemble_fore(1:60:end), '^-', 'LineWidth', 1, 'color', [255, 0, 255]./255)
% plot(time(te_index(1:60:end)), bagging_cnn_pre(1:60:end), '<-', 'LineWidth', 1, 'color', [255, 255, 0]./255)
% plot(time(te_index(1:60:end)), fusing_lstm_pre(1:60:end), '>-', 'LineWidth', 1, 'color', [0, 255, 255]./255)


% sub figure time(te_index(1)) = 00:00 8 Mar 2007
% sub figure 1  Mar 08 07:00 - 7:30
time_index1 = [(te_index(1)+7*60):1: (te_index(1)+7*60 + 0.5*60)];
value_index1 = [(7*60):1: (7*60+0.5*60)];
figure()
plot(time(time_index1), te_value(value_index1), '-', 'LineWidth', 1.5, 'color', [255, 0, 0]./255)
hold on 
plot(time(time_index1), t_static(value_index1), 'o-', 'LineWidth', 1.5,  'color', [0, 255, 0]./255)
plot(time(time_index1), t_dynamic(value_index1), 's-', 'LineWidth', 1.5, 'color', [0, 0, 255]./255)
plot(time(time_index1), ensemble_fore(value_index1), '^-', 'LineWidth', 1.5, 'color',  [142, 207, 201]./255)
plot(time(time_index1), bagging_cnn_pre(value_index1), '+-', 'LineWidth', 1.5, 'color', [250, 127, 111]./255)
plot(time(time_index1), fusing_lstm_pre(value_index1), '*-', 'LineWidth', 1.5, 'color', [190, 184, 220]./255)
xlabel('time')
ylabel('active power load')
ylim([0, 14000])
legend('true value', 'static ensemble', 'dynamic ensemble','trim aggregation ANNs', 'bagging CNNs', 'fusing LSTMs','Fontsize', 15)
set(gca,'FontSize',20);
set(gcf,'position',[5 5 600 400])
print(gcf,'test11b','-dpng','-r600')

% sub figure 2  time(te_index(1)) = 00:00 8 Mar 2007
% sub figure 2  Mar 12 08:00 - 8:30 
time_index2 = [(te_index(1)+4*24*60+8.5*60) : 1 :(te_index(1)+4*24*60+8*60 +1*60)];
value_index2 = [(4*24*60+8.5*60) : 1 :(4*24*60+8*60 +1*60)];
figure()
plot(time(time_index2), te_value(value_index2), '-', 'LineWidth', 1.5, 'color', [255, 0, 0]./255)
hold on 
plot(time(time_index2), t_static(value_index2), 'o-', 'LineWidth', 1.5,  'color', [0, 255, 0]./255)
plot(time(time_index2), t_dynamic(value_index2), 's-', 'LineWidth', 1.5, 'color', [0, 0, 255]./255)
plot(time(time_index2), ensemble_fore(value_index2), '^-', 'LineWidth', 1.5, 'color',  [142, 207, 201]./255)
plot(time(time_index2), bagging_cnn_pre(value_index2), '+-', 'LineWidth', 1.5, 'color', [250, 127, 111]./255)
plot(time(time_index2), fusing_lstm_pre(value_index2), '*-', 'LineWidth', 1.5, 'color', [190, 184, 220]./255)
ylim([0, 5000])
xlabel('time')
ylabel('active power load')
% legend('true value', 'static ensemble', 'dynamic ensemble','trim aggregation', 'bagging cnn', 'fusing lstm','Fontsize', 15)
set(gca,'FontSize',20);
set(gcf,'position',[5 5 600 400])
print(gcf,'test11c','-dpng','-r600')

% % =========================================================================
clear all
load test_case1_aep.mat

% figure()
% plot(time(te_index), te_value, '-', 'LineWidth', 1, 'color', [255, 0, 0]./255)
% hold on 
% plot(time(te_index), t_static, 'o-', 'LineWidth', 1,  'color', [0, 255, 0]./255)
% plot(time(te_index), t_dynamic, 's-', 'LineWidth', 1, 'color', [0, 0, 255]./255)
% % plot(time(te_index), ensemble_fore, '^-', 'LineWidth', 1, 'color', [255, 0, 255]./255)
% % plot(time(te_index), bagging_cnn_pre, '<-', 'LineWidth', 1, 'color', [255, 255, 0]./255)
% % plot(time(te_index), fusing_lstm_pre, '>-', 'LineWidth', 1, 'color', [0, 255, 255]./255)
% legend('true value', 'static ensemble', 'dynamic ensemble')

figure()
plot(time(te_index(1:6:end)), te_value(1:6:end), '-', 'LineWidth', 1.5, 'color', [255, 0, 0]./255)
hold on 
plot(time(te_index(1:6:end)), t_static(1:6:end), 'o-', 'LineWidth', 1.5,  'color', [0, 255, 0]./255)
plot(time(te_index(1:6:end)), t_dynamic(1:6:end), 's-', 'LineWidth', 1.5, 'color', [0, 0, 255]./255)
xlabel('time')
ylabel('appliance power load')
set(gca,'FontSize',20);
set(gcf,'position',[5 5 1600 600])
legend('true value', 'static ensemble', 'dynamic ensemble','Fontsize', 15)
print(gcf,'test12a','-dpng','-r600')
% plot(time(te_index(1:6:end)), ensemble_fore(1:6:end), '^-', 'LineWidth', 1, 'color', [255, 0, 255]./255)
% plot(time(te_index(1:6:end)), bagging_cnn_pre(1:6:end), '<-', 'LineWidth', 1, 'color', [255, 255, 0]./255)
% plot(time(te_index(1:6:end)), fusing_lstm_pre(1:6:end), '>-', 'LineWidth', 1, 'color', [0, 255, 255]./255)


% sub figure 1    time(te_index(1)) = 2016-05-11 00:00:00
% May 14 08:00 - May 14 13:00
time_index1 = [(te_index(1)+3*24*6 + 7*6):1:(te_index(1)+3*24*6 + 7*6+ 5*6)];
value_index1 = [(te_value(1)+3*24*6+ 7*6):1: (te_value(1)+3*24*6 + 7*6+ 5*6)];
figure()
plot(time(time_index1), te_value(value_index1), '-', 'LineWidth', 1.5, 'color', [255, 0, 0]./255)
hold on 
plot(time(time_index1), t_static(value_index1), 'o-', 'LineWidth', 1.5,  'color', [0, 255, 0]./255)
plot(time(time_index1), t_dynamic(value_index1), 's-', 'LineWidth', 1.5, 'color', [0, 0, 255]./255)
plot(time(time_index1), ensemble_fore(value_index1), '^-', 'LineWidth', 1.5, 'color', [142, 207, 201]./255)
plot(time(time_index1), bagging_cnn_pre(value_index1), '+-', 'LineWidth', 1.5, 'color', [250, 127, 111]./255)
plot(time(time_index1), fusing_lstm_pre(value_index1), '*-', 'LineWidth', 1.5, 'color', [190, 184, 220]./255)
xlabel('time')
ylabel('appliance power load')
ylim([0, 800])
legend('true value', 'static ensemble', 'dynamic ensemble','trim aggregation ANNs', 'bagging CNNs', 'fusing LSTMs','Fontsize', 15)
set(gca,'FontSize',20);
set(gcf,'position',[5 5 600 400])
print(gcf,'test12b','-dpng','-r600')

% sub figure 2    time(te_index(1)) = 2016-05-11 00:00:00
% May 16 12:00 - May 16 17:00
time_index1 = [(te_index(1)+5*24*6 + 12*6):1:(te_index(1)+5*24*6 + 12*6+ 5*6)];
value_index1 = [(te_value(1)+5*24*6+ 12*6):1: (te_value(1)+5*24*6 + 12*6+ 5*6)];
figure()
plot(time(time_index1), te_value(value_index1), '-', 'LineWidth', 1.5, 'color', [255, 0, 0]./255)
hold on 
plot(time(time_index1), t_static(value_index1), 'o-', 'LineWidth', 1.5,  'color', [0, 255, 0]./255)
plot(time(time_index1), t_dynamic(value_index1), 's-', 'LineWidth', 1.5, 'color', [0, 0, 255]./255)
plot(time(time_index1), ensemble_fore(value_index1), '^-', 'LineWidth', 1.5, 'color', [142, 207, 201]./255)
plot(time(time_index1), bagging_cnn_pre(value_index1), '+-', 'LineWidth', 1.5, 'color', [250, 127, 111]./255)
plot(time(time_index1), fusing_lstm_pre(value_index1), '*-', 'LineWidth', 1.5, 'color', [190, 184, 220]./255)
xlabel('time')
ylabel('appliance power load')
ylim([0, 160])
%legend('true value', 'static ensemble', 'dynamic ensemble','trim aggregation', 'bagging cnn', 'fusing lstm')
set(gca,'FontSize',20);
set(gcf,'position',[5 5 600 400])
print(gcf,'test12c','-dpng','-r600')