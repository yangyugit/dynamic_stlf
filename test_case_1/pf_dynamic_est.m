function w = pf_dynamic_est(z, r_svar, s_svar)
% dynamic estimation method based on the stat space method
% particle fiter (consideting the stability of numercial computing)
% w is the weighting vector for estimation
% r and s is the state and measurement noise covariance
particles = 200;
R = r_svar^2;

w(:,1) = z(:, 1); 

% generate particles 
pr = zeros(size(z,1), particles);

for i = 1 : particles
    pr(:,i) = z(:, 1) + normrnd(0, r_svar^3, [size(z, 1), 1]);
end

for i = 2 : size(z, 2)
    % sample (based on state equation)
    for j = 1 : particles
        xpr_(:, j) = pr(:, j) + normrnd(0, r_svar^3, [size(z, 1), 1]);
    end
    
    % compute weight (based on measurement equation)
    for j = 1 : particles
        zpr_(:, j) = xpr_(:, j) + normrnd(0, s_svar, [size(z, 1), 1]);
        weight(i, j) = sqrt((2*pi)^size(z,1))*sqrt(det(R))*...,
            exp(-0.5*(z(:,i)-zpr_(:,j))'*inv(R)*(z(:,i)-zpr_(:,j))) + 1e-99;
    end
    
    % normalize
    weight(i,:) = weight(i,:)./sum(weight(i,:));
    
    % resample
    outIndex = randomR(weight(i,:));
    pr = xpr_(:, outIndex);
    
    w(:,i) = mean(pr,2);
     
end    
end