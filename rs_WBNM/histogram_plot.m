function histogram_plot(X)
Isubdiag = find(tril(ones(length(X)),-1)); % Indexes of all the values below the diagonal
h = histogram(X(Isubdiag),'Normalization','probability');
h.NumBins = 10;
h.BinWidth = 0.1;
h.BinLimits = [0 1];
end

