clc;clear;close all;
expnum = dir(['Expfile\' '*.txt']);

for n = 1:1201
    expName = ['exp' num2str(n, '%03d')];
    simulate_model(expName);
end
