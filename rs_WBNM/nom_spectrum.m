function [f,nspectrum]=nom_spectrum(x,fs)
N = length(x);
N = 2^nextpow2(N);
f=(0:N/2)*fs/N;
spectrum = abs(fft(x-mean(x),N));
spectrum = spectrum.^2;
total = sum(spectrum(1:N/2+1)); % ����Ҷ����ֵ���жԳ���
nspectrum = spectrum/total;
nspectrum = nspectrum(1:N/2+1); % ����Ƶ��Ϊ fs,Ƶ�ʷֱ���Ϊ fs/N
end



