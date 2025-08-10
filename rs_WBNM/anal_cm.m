function [degree,Eglo,cpl,cc] = anal_cm(W)

N = size(W,1);
W_bin = weight_conversion(W, 'binarize');
% L = weight_conversion(W, 'lengths');
% D = distance_wei(L);

degree = degrees_und(W_bin);degree = sum(degree)/N;
Eglo = efficiency_bin(W_bin);
[~,cpl] = Aver_Path_Length(W_bin);% cpl = charpath(D);
cc = clustering_coef_bu(W_bin);cc = sum(cc)/N;

end


