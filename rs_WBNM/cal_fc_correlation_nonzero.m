function corr = cal_fc_correlation_nonzero(X, Y)
Isubdiag1 = find(tril(X,-1)); % Indexes of all the nonzero values of X below the diagonal
Isubdiag2 = find(tril(Y,-1)); % Indexes of all the nonzero values of Y below the diagonal
Isubdiag = intersect(Isubdiag1,Isubdiag2); % Indexes of all the nonzero values shared by X and Y below the diagonal
corr_matrix = corrcoef(X(Isubdiag), Y(Isubdiag));
corr = corr_matrix(1, 2);
end

