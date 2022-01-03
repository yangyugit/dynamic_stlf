function [rmse, mae, mape] = error_indices(pred, true)
rmse = sqrt(sum((true - pred).^2)./size(true, 2));
mae = sum(abs(true-pred))./size(true, 2);
mape = sum(abs((true - pred)./true))/size(true, 2);
end