function plv = cal_plv(eegData,range)

order = 4;
srate = 1000;
Nsample = length(eegData);
N = size(eegData,2);
plv = zeros(N);
if length(range) == 1
    [b,a] = butter(order, 2/srate*range, 'low');
elseif length(range) == 2
    [b,a] = butter(order, 2/srate*range, 'bandpass');
end
filteredData = filtfilt(b,a,eegData);

% hilbert transform
eegData_phase = angle(hilbert(filteredData));

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
