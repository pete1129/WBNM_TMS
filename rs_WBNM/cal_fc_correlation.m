function corr = cal_fc_correlation(X, Y)
Isubdiag = find(tril(ones(length(X)),-1)); % Indexes of all the values below the diagonal
corr_matrix = corrcoef(X(Isubdiag), Y(Isubdiag));
corr = corr_matrix(1, 2);
end

