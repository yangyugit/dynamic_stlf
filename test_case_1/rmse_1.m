% the function of MSE
function z = mse_1(x,y)
% x and y is the same length vector 
z = sqrt(mean(((x - y).^2)));
end