function excs = simulate_wendlingModel(Param)
P = wendModel_params('Cij',Param.Cij,'C',Param.globalcoupling,'D',Param.strength);
N = P.N;
dt = Param.dt_ms;
Nt = Param.duration_ms;
excs = zeros(N, Nt/dt+1);%68*2001
yinit = zeros(8*N, 1);
fs = 1000;
re = 1/fs;
stim = ones(N,1);
for i = 2 : length(excs)
    pt = random('Normal', 90, 30, N, 1);

    if (i >= 1001) && (i <= 2001)
        % 时间 t = r - stim_start /((r+1)*(r+1))
        r = i - 1001;  % ms 
        stim = P.D * 1000 * stim'/((r+1)*(r+1)); % Requires actual measurement using SimNIBS
    yinit = y(end,:)';
    
end
excs = excs';

end
