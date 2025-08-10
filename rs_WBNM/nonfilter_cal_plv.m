function plv = nonfilter_cal_plv(eegData)

Nsample = length(eegData);
N = size(eegData,2);
plv = zeros(N);

% hilbert transform
eegData_phase = angle(hilbert(eegData));

% calculate plv for each trial
for ra = 1:N-1
    for rb = ra+1:N
        e = exp(1i*(eegData_phase(:,rb) - eegData_phase(:,ra)));
        plv(ra, rb) = abs(sum(e))/Nsample;
        plv(rb, ra) = plv(ra,rb); 
    end
end
plv(1:N+1:end) = 1;
end
