function [f,nspectrum]=nom_spectrum(x,fs)
N = length(x);
N = 2^nextpow2(N);
f=(0:N/2)*fs/N;
spectrum = abs(fft(x-mean(x),N));
spectrum = spectrum.^2;
total = sum(spectrum(1:N/2+1)); % 傅里叶返回值具有对称性
nspectrum = spectrum/total;
nspectrum = nspectrum(1:N/2+1); % 采样频率为 fs,频率分辨率为 fs/N
end



