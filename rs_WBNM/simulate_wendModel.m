function excs = simulate_wendModel(Param)

P = wendModel_params('Cij',Param.Cij,'C',Param.globalcoupling);
% P = wendModel_params('Cij',Param.Cij,'Dij',Param.Dij,'C',Param.globalcoupling);
N = P.N;
dt = Param.dt_ms;
Nt = Param.duration_ms;
% D_ndt = cal_delay(dt, P.Dij, 3);
% startind = max(D_ndt, [], 'all') + 1;
excs = zeros(N, Nt/dt+1);
% excs = zeros(N, startind + length(Nt/dt));
yinit = zeros(8*N, 1);

for i = 2 : length(excs)
    pt = random('Normal', 90, 30, N, 1);
    [t,y] = ode45(@(t,y) wendModel_ode(t, y, pt, P, excs, i), 0:0.001:0.001, yinit);
    Nnode = y(end, 2:8:end) - y(end, 3:8:end) - y(end, 4:8:end);
    excs(:, i) = Nnode';
    yinit = y(end,:)';
end
excs = excs';
end

