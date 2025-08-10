function dy = wendModel_ode(t, y, pt, P, excs, i, stim)
    p = y(1:8:end);
    e = y(2:8:end);
    s = y(3:8:end);
    f = y(4:8:end);
    pp = y(5:8:end);
    ee = y(6:8:end);
    ss = y(7:8:end);
    ff = y(8:8:end);
    dy = zeros(8*P.N, 1);
    interafferent = zeros(P.N, 1);
   
    for n = 1 : P.N
        for m = 1 : P.N
            interafferent(n) = interafferent(n) + P.Cij(n, m) * sigmoid(excs(m, i-1));
        end
    end
    
    % reload
    dy(1:8:end) = pp;
    dy(2:8:end) = ee;
    dy(3:8:end) = ss;
    dy(4:8:end) = ff;
    dy(5:8:end) = P.A * P.a * sigmoid(e - s - f + stim) - 2 * P.a * pp - P.a ^ 2 * p;
    dy(6:8:end) = P.A * P.a * (pt + P.C_e2p * sigmoid(P.C_p2e * p) + P.C * interafferent) - 2 * P.a * ee - P.a ^ 2 * e;
    dy(7:8:end) = P.B * P.b * P.C_s2p * sigmoid(P.C_p2s * p) - 2 * P.b * ss - P.b ^ 2 * s;
    dy(8:8:end) =  P.G * P.g * P.C_f2p * sigmoid(P.C_p2f * p - s * P.C_s2f / P.C_s2p) - 2 * P.g * ff - P.g ^ 2 * f;
end
