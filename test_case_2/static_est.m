function w_v = static_est(member_model, true_line, window_size)
% static estiamtion method based on the measurement equation
% x is the fitting of member model
% y is the true value
% window_size is the number of points used to estimate

% the initial window_size-1 points of weighting vector is 1/numl(member models) i.e. 1/n
m1 = member_model(1,:);
m2 = member_model(2,:);
m3 = member_model(3,:);
tr = true_line;

w_v = zeros(size(member_model));

w_v(:, 1:(window_size-1)) = 1./size(member_model, 1);

for i =  window_size: size(member_model, 2)
    y = tr(i-window_size+1:i)';
    x = member_model(: , i-window_size+1: i)';
    w_v (:, i) = inv(x' * x) * x' * y;
end
end