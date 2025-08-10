C = 0:0.1:120;
nTrial = 50;
dt_ms = 1;
dt_sample_ms = 1000;
duration_ms = 3*1000;
RelTol = 1e-3;
AbsTol = 1e-6;

for c = 1 : length(C)
    expName =sprintf('exp%03d', c); 
    filename = ['Expfile\' expName '.txt'];
    globalcoupling = C(c);
    T = table(nTrial, dt_ms, dt_sample_ms, duration_ms, globalcoupling, RelTol, AbsTol);
    writetable(T, filename);
end


            