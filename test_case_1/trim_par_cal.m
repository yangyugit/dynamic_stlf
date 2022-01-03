% trim_par_cal
function y = trim_par_cal(rmse1, rmse2, rmse3)
a = [rmse1, rmse2, rmse3];
min_rmse = mean(a); % the mean value is used as threshold
y = find(a<min_rmse);
end