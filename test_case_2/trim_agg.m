function y = trim_agg(trim, fnn_te_pre, elm_te_pre, rbf_te_pre)
    pre =   [fnn_te_pre;
            elm_te_pre;
            rbf_te_pre];
    y = mean(pre(trim, :), 1);
end