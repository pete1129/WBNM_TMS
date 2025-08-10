%% 时频分析
% 先获取值
clc;clear;close all;
load("Data/excs.mat");
load("Data/exc.mat");
fontsize=12;
ticklabelsize = 10;
fontname='Times New Roman';
%% plv圆形图
figure
load('Data\sim_plv.mat')
labels = {'lh.bankssts', 'lh.caudalanteriorcingulate', 'lh.caudalmiddlefrontal', 'lh.cuneus', 'lh.entorhinal', 'lh.fusiform', 'lh.inferiorparietal', 'lh.inferiortemporal', 'lh.isthmuscingulate', 'lh.lateraloccipital', 'lh.lateralorbitofrontal', 'lh.lingual', 'lh.medialorbitofrontal', 'lh.middletemporal', 'lh.parahippocampal', 'lh.paracentral', 'lh.parsopercularis', 'lh.parsorbitalis', 'lh.parstriangularis', 'lh.pericalcarine', 'lh.postcentral', 'lh.posteriorcingulate', 'lh.precentral', 'lh.precuneus', 'lh.rostralanteriorcingulate', 'lh.rostralmiddlefrontal', 'lh.superiorfrontal', 'lh.superiorparietal', 'lh.superiortemporal', 'lh.supramarginal', 'lh.frontalpole', 'lh.temporalpole', 'lh.transversetemporal', 'lh.insula', 'rh.bankssts', 'rh.caudalanteriorcingulate', 'rh.caudalmiddlefrontal', 'rh.cuneus', 'rh.entorhinal', 'rh.fusiform', 'rh.inferiorparietal', 'rh.inferiortemporal', 'rh.isthmuscingulate', 'rh.lateraloccipital', 'rh.lateralorbitofrontal', 'rh.lingual', 'rh.medialorbitofrontal', 'rh.middletemporal', 'rh.parahippocampal', 'rh.paracentral', 'rh.parsopercularis', 'rh.parsorbitalis', 'rh.parstriangularis', 'rh.pericalcarine', 'rh.postcentral', 'rh.posteriorcingulate', 'rh.precentral', 'rh.precuneus', 'rh.rostralanteriorcingulate', 'rh.rostralmiddlefrontal', 'rh.superiorfrontal', 'rh.superiorparietal', 'rh.superiortemporal', 'rh.supramarginal', 'rh.frontalpole', 'rh.temporalpole', 'rh.transversetemporal', 'rh.insula'};
% c = colorbar('position',[0.85,0.22,0.04,0.6]);
%clim([0.6 1]); % 颜色映射
% c.Label.String = ('sim\_plv');
for i = 1:68
    sim_plv(i,i) = 0;
    for j =1:68
        if sim_plv(i,j) < 0.6
            sim_plv(i,j) = 0;
        end
    end
end
circularGraph(sim_plv, 'Label', labels);

title('PLV functional connection diagram (>0.5)', 'FontSize', 14);
lines = findall(gca, 'Type', 'Line');
for i = 1:length(lines)
    set(lines(i), 'LineWidth', 0.1); % 线宽设置
end
set(gcf, 'Position', [300,65,930,900]); 
saveas(gcf, 'Figure\simplv_circle.fig')
%% Pearson圆形图
figure
load('Data\sim_fc.mat')
labels = {'lh.bankssts', 'lh.caudalanteriorcingulate', 'lh.caudalmiddlefrontal', 'lh.cuneus', 'lh.entorhinal', 'lh.fusiform', 'lh.inferiorparietal', 'lh.inferiortemporal', 'lh.isthmuscingulate', 'lh.lateraloccipital', 'lh.lateralorbitofrontal', 'lh.lingual', 'lh.medialorbitofrontal', 'lh.middletemporal', 'lh.parahippocampal', 'lh.paracentral', 'lh.parsopercularis', 'lh.parsorbitalis', 'lh.parstriangularis', 'lh.pericalcarine', 'lh.postcentral', 'lh.posteriorcingulate', 'lh.precentral', 'lh.precuneus', 'lh.rostralanteriorcingulate', 'lh.rostralmiddlefrontal', 'lh.superiorfrontal', 'lh.superiorparietal', 'lh.superiortemporal', 'lh.supramarginal', 'lh.frontalpole', 'lh.temporalpole', 'lh.transversetemporal', 'lh.insula', 'rh.bankssts', 'rh.caudalanteriorcingulate', 'rh.caudalmiddlefrontal', 'rh.cuneus', 'rh.entorhinal', 'rh.fusiform', 'rh.inferiorparietal', 'rh.inferiortemporal', 'rh.isthmuscingulate', 'rh.lateraloccipital', 'rh.lateralorbitofrontal', 'rh.lingual', 'rh.medialorbitofrontal', 'rh.middletemporal', 'rh.parahippocampal', 'rh.paracentral', 'rh.parsopercularis', 'rh.parsorbitalis', 'rh.parstriangularis', 'rh.pericalcarine', 'rh.postcentral', 'rh.posteriorcingulate', 'rh.precentral', 'rh.precuneus', 'rh.rostralanteriorcingulate', 'rh.rostralmiddlefrontal', 'rh.superiorfrontal', 'rh.superiorparietal', 'rh.superiortemporal', 'rh.supramarginal', 'rh.frontalpole', 'rh.temporalpole', 'rh.transversetemporal', 'rh.insula'};
c = colorbar('position',[0.85,0.22,0.04,0.6]);
clim([0.4 0.8]); % 颜色映射
c.Label.String = ('sim\_fc');
for i = 1:68
    sim_fc(i,i) = 0;
    for j =1:68
        if sim_fc(i,j) <= 0.5
            sim_fc(i,j) = 0;
        end
    end
end
circularGraph(sim_fc, 'Label', labels);

title('Pearson functional connection diagram (>0.3)', 'FontSize', 14);
lines = findall(gca, 'Type', 'Line');
for i = 1:length(lines)
    set(lines(i), 'LineWidth', 0.1); % 线宽设置
end
set(gcf, 'Position', [300,65,930,900]); 
saveas(gcf, 'Figure\simfc_circle.fig')

%% 电位值
figure % 原值
nTrial = 30;
load("Data/excs.mat");
plot(linspace(-1000, 2001, 3001), excs, 'LineWidth', 1.5);
title('波形（刺激线圈MagVenture MC-125）');
xlabel('时间 (μs)');
ylabel('幅值（mV)');
xlim([-1000, 2001]);
ylim([-15 25]);
grid on;
hold on;
saveas(gcf, 'Figure\excs.fig')
% excs 是 EEG 数据，大小为 3001 × 68
%% 计算 GMFP
[timePoints, numChannels] = size(excs);
GMFP = zeros(timePoints, 1);

for t = 1:timePoints
    meanVoltage = mean(excs(t, :)); % 所有通道的平均值
    GMFP(t) = sqrt(sum((excs(t, :) - meanVoltage).^2) / numChannels);
end

% 绘制 GMFP 曲线
time = linspace(-1000, 2001, timePoints); % 时间轴
figure;
plot(time, GMFP, 'LineWidth', 1.5);
xlim([-500, 1001]);
ylim([1 4]);
% set(gca, 'xtick', -500:250:1000);
set(gca,'ytick',1:1:4)
xlabel('Time (ms)','FontSize',fontsize,'FontName',fontname);
ylabel('Amplitude (mV)','FontSize',fontsize,'FontName',fontname);
title('GMFP');
grid on;

% 计算TEP（对所有试次进行平均）
TEP = mean(excs, 2);  % 按照试次维度求平均，得到刺激诱发电位（TEP）

% 绘制TEP曲线
time = linspace(-1000, 2001, timePoints);  % 时间轴
figure;
plot(time, TEP, 'LineWidth', 1.5);
xlim([-500, 1001]);
xlabel('时间 (ms)');
ylabel('诱发电位 (mV)');
title('TMS诱发电位 (TEP)');
grid on;

% 选择关注的通道：通道 3, 26, 27
channels_of_interest = [3, 26, 27];

% 创建一个新的矩阵存储LMFP
LMFP = zeros(timePoints, 1);  % 用于存储所有时间点的局部场强

for t = 1:timePoints
    % 获取指定通道的电位数据
    data_at_time = excs(t, channels_of_interest);  % 当前时间点所有通道的电位

    % 计算当前时间点局部场强：该时间点的通道电位与该时间点所有通道的平均电位的差异
    meanVoltage = mean(data_at_time);  % 所有选择通道的平均电位
    LMFP(t) = sqrt(sum((data_at_time - meanVoltage).^2) / length(channels_of_interest));  % 计算局部场强
end

% 绘制LMFP曲线
time = linspace(-1000, 2001, timePoints);  % 时间轴，根据实际情况调整时间范围
figure;
plot(time, LMFP, 'LineWidth', 1.5);
xlim([-500, 1001]);
ylim([0 2]);
set(gca,'ytick',0:0.5:2)
xlabel('Time (ms)','FontSize',fontsize,'FontName',fontname);
ylabel('Amplitude (mV)','FontSize',fontsize,'FontName',fontname);
title('LMFP - Region 3, 26, 27','FontSize',fontsize,'FontName',fontname);
grid on;
%% 带状图
% figure % plot
% hold on;
% labels = {'lh.bankssts', 'lh.caudalanteriorcingulate', 'lh.caudalmiddlefrontal', 'lh.cuneus', 'lh.entorhinal', 'lh.fusiform', 'lh.inferiorparietal', 'lh.inferiortemporal', 'lh.isthmuscingulate', 'lh.lateraloccipital', 'lh.lateralorbitofrontal', 'lh.lingual', 'lh.medialorbitofrontal', 'lh.middletemporal', 'lh.parahippocampal', 'lh.paracentral', 'lh.parsopercularis', 'lh.parsorbitalis', 'lh.parstriangularis', 'lh.pericalcarine', 'lh.postcentral', 'lh.posteriorcingulate', 'lh.precentral', 'lh.precuneus', 'lh.rostralanteriorcingulate', 'lh.rostralmiddlefrontal', 'lh.superiorfrontal', 'lh.superiorparietal', 'lh.superiortemporal', 'lh.supramarginal', 'lh.frontalpole', 'lh.temporalpole', 'lh.transversetemporal', 'lh.insula', 'rh.bankssts', 'rh.caudalanteriorcingulate', 'rh.caudalmiddlefrontal', 'rh.cuneus', 'rh.entorhinal', 'rh.fusiform', 'rh.inferiorparietal', 'rh.inferiortemporal', 'rh.isthmuscingulate', 'rh.lateraloccipital', 'rh.lateralorbitofrontal', 'rh.lingual', 'rh.medialorbitofrontal', 'rh.middletemporal', 'rh.parahippocampal', 'rh.paracentral', 'rh.parsopercularis', 'rh.parsorbitalis', 'rh.parstriangularis', 'rh.pericalcarine', 'rh.postcentral', 'rh.posteriorcingulate', 'rh.precentral', 'rh.precuneus', 'rh.rostralanteriorcingulate', 'rh.rostralmiddlefrontal', 'rh.superiorfrontal', 'rh.superiorparietal', 'rh.superiortemporal', 'rh.supramarginal', 'rh.frontalpole', 'rh.temporalpole', 'rh.transversetemporal', 'rh.insula'};
% for i = 1:68
%     plot(linspace(-500, 501, 1001), excs(501:1501, i) + i * 10); % 向上偏移i*10个单位
% end
% plot([0 0],[0 700],'k--','LineWidth', 0.1)
% xlabel('Time');
% ylabel('Brain Regions');
% xlim([-500,500]);
% yticks(10:10:680);
% yticklabels(labels);
% title('Time Series of 68 Brain Regions(add tms)');
% set(gcf, 'Position', [100, 100, 1200, 800]); 
% hold off;
% saveas(gcf, 'Figure\strip.fig')

figure % plot
hold on;
% 创建一个颜色映射
colors = lines(68);  % 使用MATLAB内建的线条颜色映射（有68种颜色）
% 使用 Region1 到 Region68 作为标签
yticks(10:10:680);
yticklabels(arrayfun(@(x) sprintf('Region%d', x), 1:68, 'UniformOutput', false));
for i = 1:68
    % 绘制每条曲线，使用不同的颜色
    plot(linspace(-500, 1001, 1501), excs(501:2001, i) + i * 10, 'Color', colors(i, :), 'LineWidth', 1.5); % 向上偏移i*10个单位
end
% 添加垂直虚线
plot([0 0],[0 700],'k--','LineWidth', 0.8);
% 设置标签和标题
% xlabel('Time (ms)');
% ylabel('Brain Regions');
% title('Potential(mV)');
% 设置图形的 x 和 y 轴范围
xlim([-500,1000]);
% 优化图形的大小和细长的比例
set(gcf, 'Position', [100, 100, 100, 600]);
% 去掉 label 显示
set(gca, 'yticklabel', []);
% 显示图形
hold off;
% 保存图形
saveas(gcf, 'Figure/strip_optimized.fig');
%% 时域分析（STFT）
i = 3;
new_time = linspace(-500,501,1001);
N = numel(excs(501:1501,i)); % the number of time points
fs = 1000;
    nfft = 2^nextpow2(N); % the number of FFT points
    winsize = round([0.1 0.2 0.3]*fs); % three window sizes                                                                                                         .2s, 0.4s, 0.6s) are used in STFT for comparison
    for n = 1:numel(winsize)
        [P(:,:,n),f] = subfunc_stft(excs(501:1501,i), winsize(n), nfft, fs);
    end
f_lim = [0, 41]; % specify the frequency range  
% to be shown (remove 0Hz)
f_idx = find((f<=f_lim(2))&(f>=f_lim(1)));
t_lim = 1000; % specify the time range to be shown
t_idx = new_time;

figure('units','normalized','position',[0.1    0.15    0.8    0.7])
subplot(2,2,1)
hold on; box on
plot(excs(1001:2001,i),'linewidth',1);
plot([0 1000],[0 0],'k--','LineWidth', 0.1)
plot([0 0],[0 18],'k--','LineWidth', 0.1)
set(gca,'xlim',[0,1000])
xlabel('Time (ms)')
ylabel('Amplitude (mV)')
title('wendlingmodel(add tms)','fontsize',12)
for n = 1:numel(winsize)
    subplot(2,2,n+1)
    imagesc(t_idx,f(f_idx),P(f_idx, 1:1001,n))
    colormap(jet);
    xlabel('Time (ms)')
    ylabel('Frequency (Hz)')
    set(gca,'xlim',[-500 500],'ylim',f_lim)
    axis xy; hold on;
    %plot([1000,0],[1000 20],'w--','LineWidth', 0.1)
    text(1000,f_lim(2)/2,'Power (dB)','rotation',90,'horizontalalignment','center','verticalalignment','top')
    title(['Spectrogram (winsize = ',num2str(winsize(n)/fs,'%1.2g'),'s)'],'fontsize',12)
    colorbar
end
saveas(gcf, 'Figure\stft.fig')
%% 去趋势化
figure 
plot(detrend(excs(:,:)))
set(gca,'xlim',[1001 3000])
saveas(gcf, 'Figure\excs_detr.fig')
%% 功率谱密度
figure('units','normalized')
psd_matrix = [];
fs = 1000;
for i = 1:68
    [pxx, f] = pwelch(detrend(excs(1001:2001,i)), [], [], [], fs);
    psd_matrix = [psd_matrix; pxx'];
end
f_lim = f((f>=0)&(f<=50));
mean_psd = mean(psd_matrix, 1000);
%plot(f, 10*log10(mean_psd)); % 转换为dB
plot(f, mean_psd); % 转换为dB
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Overall Power Spectral Density');
set(gca,'xlim',[min(f_lim),max(f_lim)])
grid on;
saveas(gcf, 'Figure\psd.fig')
%% 功率谱密度（周期图与welch图）
N = length(excs); % the number of samples (N=15000)
excs = detrend(excs); % remove the low-frequency trend from EEG
fs = 1000;
for i = 1:68
    nfft = 2^nextpow2(N); % the number of FFT points
    [P_per, f] = periodogram(excs(:,i),[],nfft,fs); % periodogram is also estimated for comparison
    [P_wel_1, f] = pwelch(excs(1001:2001,i),fs,fs/2,nfft,fs);
    [P_wel_2, f] = pwelch(excs(1001:2001,i),fs,0,nfft,fs);
    [P_wel_3, f] = pwelch(excs(1001:2001,i),fs/2,0,nfft,fs);
end

f_lim = f((f>0)&(f<=50)); % specify the frequency range to be shown

figure('units','normalized','position',[0.1    0.3    0.8    0.5])
subplot(2,2,1) 
hold on; box on;
plot(f,P_per,'k','linewidth',1) % show the periodogram in a linear scale
xlabel('Frequency (Hz)'); ylabel('Power (mV^2/Hz)')
title('Periodogram (in a linear scale)','fontsize',12)
set(gca,'xlim',[min(f_lim),max(f_lim)])
grid on;

subplot(2,2,2)
hold on; box on; 
plot(f,10*log10(P_per),'k','linewidth',1) % show the periodogram in a log scale
xlabel('Frequency (Hz)'); ylabel('Power (dB)')
title('Periodogram (in a logarithmic scale)','fontsize',12)
set(gca,'xlim',[min(f_lim),max(f_lim)])
grid on;

subplot(2,2,3)
hold on; box on;
plot(f,10*log10(P_per),'k','linewidth',0.5)
plot(f,10*log10(P_wel_1),'r','linewidth',2)
xlabel('Frequency (Hz)'); ylabel('Power (dB)')
hl = legend('Periodogram','Welch''s method (M=160, D=80)');
set(hl,'box','off','location','southwest')
set(gca,'xlim',[min(f_lim),max(f_lim)])
grid on;

subplot(2,2,4)
hold on; box on;
plot(f,P_wel_1,'r','linewidth',2)
plot(f,P_wel_2,'g','linewidth',1)
plot(f,P_wel_3,'b','linewidth',1)
xlabel('Frequency (Hz)'); ylabel('Power (dB)')
hl = legend('Welch''s (M=160, D=80)','Welch''s (M=160, D=0)','Welch''s (M=80, D=0)');
set(hl,'box','off','location','southwest')
set(gca,'xlim',[min(f_lim),max(f_lim)])
grid on;
saveas(gcf, 'Figure\pwelch.fig')
%% 熵值
load("Data/excs.mat");
entropy_stim = zeros(1,68); 
entropy_no = zeros(1,68); 
for i = 1:68
    % 计算信号的概率分布
    signal_stim = excs(1001:2001,i);
    sorted_signal_stim = sort(signal_stim); % 对信号进行排序
    signal_pdf_stim = histcounts(sorted_signal_stim, 'Normalization', 'probability');
    signal_pdf_stim(signal_pdf_stim == 0) = [];
    entropy_stim(i) = -sum(signal_pdf_stim .* log2(signal_pdf_stim)); % 刺激状态熵
end
figure
plot(entropy_stim, 'bo', 'DisplayName', '加刺激');
hold on;
load("Data/exc.mat");
for i = 1:68
    % 计算信号的概率分布
    signal_no = exc(1001:2001,i);
    sorted_signal_no = sort(signal_no); % 对信号进行排序
    signal_pdf_no = histcounts(sorted_signal_no, 'Normalization', 'probability');
    signal_pdf_no(signal_pdf_no == 0) = [];
    entropy_no(i) = -sum(signal_pdf_no .* log2(signal_pdf_no)); % 刺激状态熵
end
plot(entropy_no, 'ro', 'DisplayName', '静息态');
xlabel('通道');
ylabel('熵值');
% 将总体熵值加到标题中
title('熵值比较（排序熵）');
legend('Location', 'southeast');
xlim([1 68]);
grid on;
% 计算整体的熵
overall_entropy = mean(entropy_stim);
mean_entropy = mean(entropy_no);
disp(['Overall Stim Entropy: ', num2str(overall_entropy)]);
disp(['Overall Entropy: ', num2str(mean_entropy)]);
saveas(gcf, 'Figure\entropy.fig')
%% 功率谱密度（多窗口法）
N = length(excs); % the number of samples (N=15000)
excs = detrend(excs); % remove the low-frequency trend from EEG
fs = 1000;
for i = 1:68
    nfft = 2^nextpow2(N); % the number of FFT points
    [P_per, f] = periodogram(excs(1001:2001,i),[],nfft,fs); % periodogram is also estimated for comparison
   % check the help file to learn how to specify parameters in "pmtm.m"
    % three parameter settings are used below
    [P_mtm_1, f] = pmtm(excs(1001:2001,i),4,nfft,fs);
    [P_mtm_2, f] = pmtm(excs(1001:2001,i),2,nfft,fs);
    [P_mtm_3, f] = pmtm(excs(1001:2001,i),6,nfft,fs);
end

f_lim = f((f>0)&(f<=50)); % specify the frequency range to be shown

figure('units','normalized','position',[0.1    0.3    0.8    0.5])
subplot(1,2,1)
hold on; box on;
plot(f,10*log10(P_per),'k','linewidth',0.5)
plot(f,10*log10(P_mtm_1),'r','linewidth',2)
xlabel('Frequency (Hz)'); ylabel('Power (dB)')
hl = legend('Periodogram','Multitaper (L=4)');
set(hl,'box','off','location','southwest')
set(gca,'xlim',[min(f_lim),max(f_lim)])
grid on;

subplot(1,2,2)
hold on; box on;
plot(f,10*log10(P_mtm_1),'r','linewidth',2)
plot(f,10*log10(P_mtm_2),'g','linewidth',1)
plot(f,10*log10(P_mtm_3),'b','linewidth',1)
xlabel('Frequency (Hz)'); ylabel('Power (dB)')
hl = legend('Multitaper (L=4)','Multitaper (L=2)','Multitaper (L=6)');
set(hl,'box','off','location','southwest')
set(gca,'xlim',[min(f_lim),max(f_lim)])
grid on;
saveas(gcf, 'Figure\multitaper.fig')
%% 功率谱密度 AR模型（Yule Walker法）
N = length(excs); % the number of samples (N=15000)
excs = detrend(excs); % remove the low-frequency trend from EEG
fs = 1000;
for i = 1:68
    nfft = 2^nextpow2(N); % the number of FFT points
    [P_per, f] = periodogram(excs(1001:2001,i),[],nfft,fs); % periodogram is also estimated for comparison
   % check the help file to learn how to specify parameters in "pmtm.m"
    % three parameter settings are used below
    p1 = 20;
    p2 = 10;
    p3 = 50;
    [P_ar_1,f] = pyulear(excs(1001:2001,i),p1,nfft,fs);
    [P_ar_2,f] = pyulear(excs(1001:2001,i),p2,nfft,fs);
    [P_ar_3,f] = pyulear(excs(1001:2001,i),p3,nfft,fs);

end

f_lim = f((f>0)&(f<=50)); % specify the frequency range to be shown

figure('units','normalized','position',[0.1    0.3    0.8    0.5])
subplot(1,2,1)
hold on; box on;
plot(f,10*log10(P_per),'k','linewidth',0.5)
plot(f,10*log10(P_ar_1),'r','linewidth',2)
xlabel('Frequency (Hz)'); ylabel('Power (dB)')
hl = legend('Periodogram','AR (P=20)');
set(hl,'box','off','location','southwest')
set(gca,'xlim',[min(f_lim),max(f_lim)])

subplot(1,2,2)
hold on; box on;
plot(f,10*log10(P_ar_1),'r','linewidth',2)
plot(f,10*log10(P_ar_2),'g','linewidth',1)
plot(f,10*log10(P_ar_3),'b','linewidth',1)
xlabel('Frequency (Hz)'); ylabel('Power (dB)')
hl = legend('AR (P=20)','AR (P=10)','AR (P=50)');
set(hl,'box','off','location','southwest')
set(gca,'xlim',[min(f_lim),max(f_lim)])
saveas(gcf, 'Figure\yule_walker.fig')
%% 电场强度
v = [-0.63296498 0.078606064 -0.557224297 -0.498290991 -0.039035892 -0.357744596 -0.669545824 -0.364525044 -0.395455342 -0.519558343 0.439368094 -0.400317257 0.704769636 -0.419759061 -0.260231379 -0.549154452 -0.248249085 0.503646978 0.203623371 -0.454348245 -0.757637654 -0.358270819 -0.66771314 -0.542633891 0.578669511 0.368895491 0.113501023 -0.700784601 -0.455707004 -0.75673172 1.298077178 0.080938394 -0.540351436 -0.238353042 -7.31E-02 0.285786439 0.112837395 -0.448522568 0.162842069 -0.06580896 -3.25E-01 0.053590867 -0.351524683 -0.318728071 0.582303862 -0.300217127 0.779174241 0.073879588 -0.045496258 -4.67E-01 0.29354945 0.649338975 0.557911242 -0.386775927 -0.235836447 -0.279994553 -0.119471067 -0.481617362 0.721984094 0.743159952 0.361153114 -0.497264011 0.151428883 -0.16132528 1.233959808 0.338988843 0.01975529 0.178911981]';
E_magn = [0.148000421 0.154235634 0.661395873 0.058710149 ,0.049254986 0.084336632 0.114831768 0.107898608 0.055595344 0.074902101 0.125370908 0.060295074 0.121543746 0.156832225 0.073867155 0.124680595 0.313129228 0.21616768 0.154782609 0.05765598 0.349146179 0.081375003 0.423315371 0.083980752 0.116002826 0.561704965 0.337143542 0.163241832 0.197092725 0.202143717 0.250792653 0.068841337 0.159656376 0.145750001 0.078747789 0.100415966 0.233541536 0.052489426 0.054835822 0.05124765 0.084124615 0.062989623 0.042505289 0.055181586 0.163234786 0.047880226 0.164977253 0.076729905 0.049761568 0.104936408 0.10556583 0.156870561 0.157609206 0.049421426 0.160218913 0.041177736 0.191439009 0.073782802 0.140279519 0.210228456 0.228721375 0.113075394 0.094281915 0.115153438 0.225067321 0.07104811 0.076715574 0.077324014]';
% 可视化：柱状图
figure;
bar(v, 'FaceColor', [0.2, 0.4, 0.6]); % 设置柱状图颜色
xlabel('脑区');
ylabel('电势值 (V)');
title('68个ROI脑区的电势值分布');
grid on;
%% 扰动电压
stim = zeros(68, 1);
for i = 2 : 3001
    if (i >= 1001) && (i <= 3001)
        % 时间 t = r - stim_start /((r+1)*(r+1))
        r = i - 1001;  % 时间单位为毫秒 
        % stim = 0;
        % stim = 1000 * [-0.5707279521439742 -0.006598619669648519 -0.5986497321964788 -0.4551843527270387 -0.06352277408231169 -0.33057401033086026 -0.6012825543379874 -0.33766116235928595 -0.3817274380570266 -0.46329985307981375 0.35677234203100944 -0.36828099644590134 0.627090668080068 -0.3913962622500449 -0.253221059010044 -0.5447310254090125 -0.29743147141806536 0.38654574708394573 0.09438335603688049 -0.41537315173273454 -0.7101359229042475 -0.3778024736882859 -0.6537597384985679 -0.5107574132407685 0.4873802009989784 0.23625084790606798 0.005119798632160695 -0.6432887838062291 -0.4338877245133621 -0.6867381136552057 1.2029157269632273 0.029751406088484995 -0.5055712594686477 -0.2589244192138593 -0.09927016921455677 0.1896985477890878 0.019938838448486255 -0.41496201743252603 0.12373021339993924 -0.07924879673852872 -0.3218421630136588 0.025796455538534234 -0.3456445455497191 -0.29910484541729154 0.513095038614924 -0.2844957611526767 0.7017762786401024 0.04049645149450887 -0.0651671367549017 -0.4805968470417361 0.22597238443891393 0.5770180774527519 0.4822534983375494 -0.3596964734450596 -0.26748099608416687 -0.3095133233889734 -0.1715441166545541 -0.4614899575959378 0.6329260410947576 0.6548777251196158 0.252766217108886 -0.47811207380030724 0.10605514495495057 -0.18984525750519626 1.1517533714921564 0.28497722400185166 -0.018523824236028753 0.12676599362726548]'/((r+1)*(r+1));%F3 80%MT
        stim = 1000 * [-0.6329649803253155 0.07860606354617607 -0.5572242968315415 -0.4982909906982216 -0.03903589153411798 -0.3577445964915384 -0.6695458244680846 -0.3645250440617907 -0.39545534212154027 -0.5195583425261083 0.4393680941785612 -0.4003172573984795 0.7047696361523778 -0.4197590606572654 -0.2602313788374448 -0.5491544522527032 -0.24824908461724335 0.5036469778304183 0.2036233708929473 -0.454348245063421 -0.7576376542495499 -0.35827081917544185 -0.6677131402185874 -0.5426338909555408 0.5786695107935084 0.368895490581486 0.11350102323121963 -0.7007846011944409 -0.455707004496398 -0.7567317197533597 1.2980771783405172 0.08093839394009632 -0.540351436270583 -0.2383530424200298 -0.07313015549579906 0.2857864390428272 0.112837394835416 -0.4485225679201288 0.16284206857141795 -0.06580895981550217 -0.32512765830975726 0.05359086724931079 -0.35152468284312893 -0.3187280711108122 0.5823038622153789 -0.30021712656394645 0.7791742408608346 0.07387958816364547 -0.04549625805278425 -0.467093065236639 0.2935494502710125 0.6493389751921012 0.557911242227293 -0.38677592679015116 -0.2358364473591113 -0.27999455336391765 -0.11947106657263094 -0.4816173623553229 0.7219840939067182 0.7431599522969903 0.36115311390667765 -0.49726401051285146 0.15142888306600794 -0.16132527991790616 1.2339598076821703 0.3389888431446986 0.01975529041073931 0.17891198066993774]'/((r+1)*(r+1));%F3 MT
        end
    %pulse_array(i) = pulse; 
    stim_array(i,:) = stim;
end
save('Data/stim.mat', 'stim_array');
%% 扰动电压
load("stim.mat");  % stim_array: 3001 x 68

% 原始时间轴：0~3000ms → 映射为 -1000~2000ms
t = 0:3000;
t_redefined = t - 1000;

% 构建新的时间轴：在 -1ms 与 0ms 之间添加一个接近0的时间点（如 -1e-3）
t_insert = -1e-3;  % 接近 0ms 的时间点
t_new = [t_redefined(1:1000), t_insert, t_redefined(1001:end)];

% 构建新的电压矩阵：在每个ROI数据第1001行（0ms）前插入一行0
stim_array_mv = stim_array / 1000;  % 单位换算为 V
stim_array_new = [stim_array_mv(1:1000, :); zeros(1, 68); stim_array_mv(1001:end, :)];

% 绘图
figure;
hold on;
for i = 1:68
    plot(t_new, stim_array_new(:, i), 'LineWidth', 1.5);
end
hold off;

xlim([-10, 50]);  % 关注 -10ms ~ 50ms
xlabel('time (ms)');
ylabel('Amplitude (V)');
title('Disturbance Voltage');
%% 图论（中版）
fontsize=10;
ticklabelsize = 10;
fontname='宋体';
figure % plv模拟点节点数
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_plv_origin_nodenum,'b-o',threshold,sim_plv_nodenum_80,'k-x',threshold,emp_plv_nodenum,'r-*')
set(gca,'xlim',[0 0.5])
set(gca,'xtick',0:0.1:0.5)
set(gca,'ylim',[50 70])
set(gca,'ytick',50:2:70)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('节点数 N','FontSize',fontsize,'FontName',fontname)
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_node_num.fig')

figure % plv模拟网络和真实网络的平均度
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_degree_origin,'b-o',threshold,sim_degree,'k-x',threshold,emp_degree,'r-*')
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('平均度 D','FontSize',fontsize,'FontName',fontname)
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_degree.fig')

figure % plv模拟点特征路径长度
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_cpl_origin,'b-o',threshold,sim_cpl,'k-x',threshold,emp_cpl,'r-*')
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('特征路径长度 L','FontSize',fontsize,'FontName',fontname)
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_cpl.fig')

figure % plv模拟点平均聚类系数
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_cc_origin,'b-o',threshold,sim_cc,'k-x',threshold,emp_cc,'r-*')
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('聚类系数 CC','FontSize',fontsize,'FontName',fontname)
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_cc.fig')

figure % plv模拟点全局效率
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_Eglo_origin,'b-o',threshold,sim_Eglo,'k-x',threshold,emp_Eglo,'r-*')
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('全局效率 E','FontSize',fontsize,'FontName',fontname)
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_Eglo.fig')

figure % plv网络及随机网络的平均聚类系数
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_cc_origin,'b-o',threshold,sim_cc_origin_rand,'b--o')
hold on
plot(threshold,sim_cc,'k-x',threshold,sim_cc_rand_80,'k--x')
hold on
plot(threshold,emp_cc,'r-*',threshold,emp_cc_rand,'r--*')
hold off
legend({'模拟脑网络','模拟随机脑网络','模拟脑网络(施加tms)','模拟随机脑网络(施加tms)','真实脑网络','真实随机脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('聚类系数 CC','FontSize',fontsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_cc_rand.fig')

figure % plv网络及随机网络的特征路径长度
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_cpl_origin,'b-o',threshold,sim_cpl_origin_rand,'b--o')
hold on
plot(threshold,sim_cpl,'k-x',threshold,sim_cpl_rand_80,'k--x')
hold on
plot(threshold,emp_cpl,'r-*',threshold,emp_cpl_rand,'r--*')
hold off
legend({'模拟脑网络','模拟随机脑网络','模拟脑网络(施加tms)','模拟随机脑网络(施加tms)','真实脑网络','真实随机脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('特征路径长度 L','FontSize',fontsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_cpl_rand.fig')

figure % plv网络及随机网络的平均聚类系数比率
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_cc_origin./sim_cc_origin_rand,'b-o',threshold,sim_cc./sim_cc_rand_80,'k-x',threshold,emp_cc./emp_cc_rand,'r-*')
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('聚类系数比率 γ','FontSize',fontsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_cc_rate.fig')

figure % plv网络及随机网络的特征路径长度比率
load('Data\graph_theoretical_plv.mat')
plot(threshold,sim_cpl_origin./sim_cpl_origin_rand,'b-o',threshold,sim_cpl./sim_cpl_rand_80,'k-x',threshold,emp_cpl./emp_cpl_rand,'r-*')
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'ylim',[0.9 1.05])
set(gca,'ytick',0.9:0.05:1.05)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('特征路径长度比率 λ','FontSize',fontsize,'FontName',fontname)
legend('boxoff')
saveas(gcf, 'Figure\plv_cpl_rate.fig')

figure % plv不同阈值下模拟网络和真实网络的小世界系数
load('Data\graph_theoretical_plv.mat')
sim_cc_rate=sim_cc./sim_cc_rand_80;emp_cc_rate=emp_cc./emp_cc_rand;
sim_cpl_rate=sim_cpl./sim_cpl_rand_80;emp_cpl_rate=emp_cpl./emp_cpl_rand;
sim_cc_origin_rate=sim_cc_origin./sim_cc_origin_rand;sim_cpl_origin_rate=sim_cpl_origin./sim_cpl_origin_rand;
plot(threshold(1:22),sim_cc_origin_rate(1:22)./sim_cpl_origin_rate(1:22),'b-o',threshold(1:22),sim_cc_rate(1:22)./sim_cpl_rate(1:22),'k-x',threshold,emp_cc_rate./emp_cpl_rate,'r-*')
legend({'模拟脑网络','模拟脑网络(施加tms)','真实脑网络'},'FontSize',ticklabelsize,'FontName',fontname)
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'ylim',[1 1.2])
set(gca,'ytick',1:0.05:1.2)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('小世界系数','FontSize',fontsize,'FontName',fontname)
saveas(gcf, 'Figure\small_world_index.fig')

figure % plv不同阈值下模拟网络和真实网络的相关性变化
load('Data\phase_locking_value_thre.mat')
%plot(threshold,phase_locking_value_thre,'r-o')
%hold on
plot(threshold,phase_locking_value_origin_thre,'b-o')
hold off
set(gca,'xlim',[0 0.24])
set(gca,'xtick',0:0.04:0.24)
set(gca,'ylim',[0.67 0.73])
set(gca,'ytick',0.67:0.01:0.73)
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('阈值','FontSize',fontsize,'FontName',fontname)
ylabel('Pearson相关系数','FontSize',fontsize,'FontName',fontname)
saveas(gcf, 'Figure\plv_correlation_thre.fig')
%% LMFP
load('Data\sim_excs.mat')
fontsize=12;
ticklabelsize = 10;
fontname='Times New Roman';
% 选择关注的通道：通道 3, 26, 27
channels_of_interest = [9,11,13,15,22,24,25];
timePoints = 3001;
% 刺激强度范围
nStim = 5;
stim_range = 80:20:120;
% 创建一个新的矩阵存储 LMFP
LMFP = zeros(timePoints, nStim);  

for n = 1:2:5
    % 读取当前刺激强度的电位数据
    for t = 1:timePoints
        % 获取指定通道的电位数据
        data_at_time = sim_excs(t, channels_of_interest, 10*n-9);  % 第 n + 79 对应 80, 90, 100, 110, 120
        % 计算当前时间点局部场强
        meanVoltage = mean(data_at_time);  % 所有选择通道的平均电位
        LMFP(t, n) = sqrt(sum((data_at_time - meanVoltage).^2) / length(channels_of_interest));  % 计算局部场强
    end
end

% 绘制 LMFP 曲线
time = linspace(-1000, 2001, timePoints);  % 时间轴，根据实际情况调整时间范围
figure;
hold on;
colors = lines(nStim); % 自动生成与刺激强度数量对应的颜色

for n = 1:nStim
    plot(time, LMFP(:, n), 'Color', colors(n, :), 'LineWidth', 1.5);
end

xlim([-500, 1001]);
ylim([0 5]);
set(gca,'ytick',0:1:5)
xlabel('Time (ms)','FontSize',fontsize,'FontName',fontname);
ylabel('Amplitude (mV)','FontSize',fontsize,'FontName',fontname);
title('LMFP - Region 3, 26, 27','FontSize',fontsize,'FontName',fontname);

% 添加图例
legendStrings = arrayfun(@(x) [num2str(x) '% MT'], stim_range, 'UniformOutput', false);
legend(legendStrings, 'FontSize', 10, 'Location', 'northeast');

hold off;

%% 线性相关（这个是原图代码）
% 加载数据
load('Data\sim_plv.mat'); % 68x68 矩阵
load('Data\emp_plv.mat'); % 68x68 矩阵
figure;
fontsize = 12;
ticklabelsize = 10;
fontname = 'Times New Roman';

% 绘制散点图
scatter_plot(sim_plv, emp_plv);
hold on;

% 计算核密度估计
[x, y] = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100));
density = ksdensity([sim_plv(:), emp_plv(:)], [x(:), y(:)]);
density = reshape(density, size(x)); % 变换为网格格式

% **叠加核密度图**
contourf(x, y, density, 10, 'LineStyle', 'none', 'FaceAlpha', 0.5);

% 颜色映射（sky）
colormap(pink);
colorbar;

% 轴标签及格式
set(gca, 'xlim', [0 1], 'xtick', 0:0.1:1);
set(gca, 'ylim', [0 1], 'ytick', 0:0.1:1);
set(gca, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Simulated', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Empirical', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');

% 保存
saveas(gcf, 'Figure\simplv_empplv.fig');

hold off;

% 将矩阵转换为列向量
x = sim_plv(:);
y = emp_plv(:);

% 1. simplv 的核密度图（横长竖短）
figure;
kde_x = fitdist(x, 'Kernel'); % 核密度估计
x_range = linspace(0, 1, 100); % X 轴范围
pdf_x = pdf(kde_x, x_range); % 计算密度

plot(x_range, pdf_x, 'k', 'LineWidth', 1.5);
xlim([0,1]);
ylim([0,4]);
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman', 'FontWeight', 'bold');

% 设置图的尺寸（横长竖短）
set(gcf, 'Position', [100, 100, 400, 150]); 

saveas(gcf, 'Figure\simplv_density.fig'); % 保存图像

% 2. empplv 的核密度图（竖长横短）
figure;
kde_y = fitdist(y, 'Kernel'); % 核密度估计
y_range = linspace(0, 1, 100); % Y 轴范围
pdf_y = pdf(kde_y, y_range); % 计算密度

plot(pdf_y, y_range, 'k', 'LineWidth', 1.5);
xlim([0,4]);
ylim([0,1]);
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman', 'FontWeight', 'bold');

% 设置图的尺寸（竖长横短）
set(gcf, 'Position', [100, 100, 150, 400]); 

saveas(gcf, 'Figure\empplv_density.fig'); % 保存图像

%% 节点强度
load('Data\graph_theoretical_plv.mat')
% 将强度分为高、中、低三档
low_threshold = prctile(strength, 33); % 低档阈值（33%分位数）
high_threshold = prctile(strength, 66); % 高档阈值（66%分位数）

% 分类
StrengthLevel = cell(size(strength));
StrengthLevel(strength <= low_threshold) = {'Low'};
StrengthLevel(strength > low_threshold & strength <= high_threshold) = {'Medium'};
StrengthLevel(strength > high_threshold) = {'High'};

% 为不同档次分配颜色
colors = zeros(length(strength), 3); % 初始化颜色矩阵
colors(strcmp(StrengthLevel, 'Low'), :) = repmat([1, 0, 0], sum(strcmp(StrengthLevel, 'Low')), 1); % 低档为红色
colors(strcmp(StrengthLevel, 'Medium'), :) = repmat([0, 1, 0], sum(strcmp(StrengthLevel, 'Medium')), 1); % 中档为绿色
colors(strcmp(StrengthLevel, 'High'), :) = repmat([0, 0, 1], sum(strcmp(StrengthLevel, 'High')), 1); % 高档为蓝色

% 绘制条形图
figure;
hold on;
bar(find(strcmp(StrengthLevel, 'Low')), strength(strcmp(StrengthLevel, 'Low')), 'FaceColor', 'r');
bar(find(strcmp(StrengthLevel, 'Medium')), strength(strcmp(StrengthLevel, 'Medium')), 'FaceColor', 'g');
bar(find(strcmp(StrengthLevel, 'High')), strength(strcmp(StrengthLevel, 'High')), 'FaceColor', 'b');
hold off;

xlabel('Brain Region');
ylabel('Node Strength');
title('Node Strength of 68 Brain Regions');
legend('Low', 'Medium', 'High');
set(gca,'FontSize',ticklabelsize,'FontName',fontname)
xlabel('Region','FontSize',fontsize,'FontName',fontname)
ylabel('Node Strength','FontSize',fontsize,'FontName',fontname)
saveas(gcf, 'Figure\node_strength.fig')
%% plv模拟点时间序列
% for n = 80:10:120
for n = 204
    figure
    % figure('units','normalized','position',[0.3 0.4 0.35 0.1])
    set(gcf, 'Position', [300,100,350,250], 'Renderer', 'painters');
    load('Data\sim_excs.mat')
    fontsize=12;
    ticklabelsize = 10;
    fontname='Times New Roman';
    % plot(1000*(t-0.5),sim_excs(501:2001,[27 30 31 54],n-79),'LineWidth', 1)
    plot(1000*(t-0.5),sim_excs(501:2001,[3 15 26 27 31 54],n),'LineWidth', 1.5)
    % plot(t,sim_excs(501:1501,:,n-79),'LineWidth', 1)
    hold on;
    plot([0 0],[0 17],'k--','LineWidth', 0.1)
    set(gca,'xlim',[-500 1000])
    set(gca,'xtick',-500:250:1000)
    set(gca,'ylim',[0 17])
    set(gca,'ytick',0:2:17)
    set(gca,'FontSize',ticklabelsize,'FontName',fontname,'FontWeight', 'bold')
    xlabel('Time(ms)','FontSize',fontsize,'FontName',fontname,'FontWeight', 'bold')
    ylabel('Potential(mV)','FontSize',fontsize,'FontName',fontname,'FontWeight', 'bold')
    % title('Potential', 'FontSize', fontsize,'FontWeight', 'bold');
    % legend({'Region 3','Region 15','Region 27','Region 30','Region 31','Region 54'},'FontSize',ticklabelsize,'FontName',fontname)
    % legend('boxoff')
    hold off;
end
% title(['D = ',num2str(D(2),'%.1f')],'FontSize',fontsize,'FontName',fontname)
% saveas(gcf, 'Figure\plv_27and31and51_ts.fig')
%% plv模拟点归一化频谱或功率谱密度
ROI = [3, 15, 26, 27, 31, 54]; % 选择的ROI脑区
colors = lines(length(ROI)); % 生成不同颜色用于区分ROI
fontsize = 12;
ticklabelsize = 10;
fontname = 'Times New Roman';

load('Data\sim_ns.mat') % 加载PSD数据

% for n = 80:10:120
for n = 204
    figure;
    set(gcf, 'Position', [300,100,350,250], 'Renderer', 'painters');
    hold on;
    
    for r = 1:length(ROI)
        % 提取特定刺激强度和ROI下的PSD
        % psd_data = sim_ns(:, ROI(r), n - 79);
        psd_data = sim_ns(:, ROI(r), n);
        plot(f, psd_data, 'Color', colors(r, :), 'LineWidth', 1.5);
    end
    
    % 格式设置
    set(gca, 'xlim', [0 30]);
    set(gca, 'xtick', 0:5:30);
    set(gca, 'ylim', [0 1]);
    set(gca, 'ytick', 0:0.1:1);
    set(gca, 'FontSize', ticklabelsize, 'FontName', fontname,'FontWeight', 'bold');
    
    xlabel('Frequency (Hz)', 'FontSize', fontsize, 'FontName', fontname,'FontWeight', 'bold');
    ylabel('PSD (dB/Hz)', 'FontSize', fontsize, 'FontName', fontname,'FontWeight', 'bold');
    
    % 添加图例（标注不同ROI）
    % legend_labels = arrayfun(@(x) sprintf('Region %d', x), ROI, 'UniformOutput', false);
    % legend(legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname);
    % legend('boxoff');
    
    % 标题（标注刺激强度）
    % title(sprintf('PSD under Stimulation Intensity D = %d', n), 'FontSize', fontsize, 'FontName', fontname);
    
    hold off;
    
    % 保存图片（根据刺激强度区分）
    % saveas(gcf, sprintf('Figure\\plv_D%d_ns_multiple.fig', n));
end

%% 相关性对比图
fontsize=14;
ticklabelsize = 14;
fontname='Times New Roman';
% 定义线圈名称
coil_names = {'F3', 'F4', 'C3', 'T3', 'O1'};
% coil_names = {'Magstim D70 P-A', 'Magstim D70 A-P','Deymed 70BF','Deymed 50BF','MagVenture Cool-B65','Magstim DCC'};

% 定义刺激强度
stim_range = 80:10:120;
nStim = length(stim_range);
nCoils = length(coil_names);

% 数据
% mean_values = [0.3814, 0.3842, 0.3860, 0.3753, 0.3758; % Magstim D70 F3
               % 0.3890, 0.3905, 0.3854, 0.3812, 0.3931; % F4
               % 0.3865, 0.3895, 0.3829, 0.3880, 0.3850; % C3
               % 0.3785, 0.3828, 0.3749, 0.3829, 0.3799; % T3
               % 0.3681, 0.3639, 0.3612, 0.3596, 0.3638]; % O1
               % 0.3796, 0.3836, 0.3746, 0.3704, 0.3716]; % 70mm
% mean_values = [0.4189, 0.4421, 0.4328, 0.4083, 0.4282; %F3-DMN
%                0.4466, 0.4509, 0.4347, 0.4671, 0.4836; %F4-DMN
%                0.4262, 0.4367, 0.4299, 0.4420, 0.4347; %C3-DMN
%                0.4103, 0.4163, 0.4143, 0.4328, 0.4139; %T3-DMN
%                0.3973, 0.4010, 0.3995, 0.3981, 0.3969]; %O1-DMN
mean_values = [0.4680, 0.4893, 0.4768, 0.4806, 0.4530; %F3-DAN
               0.4694, 0.4960, 0.4824, 0.4609, 0.4836; %F4-DAN
               0.4842, 0.4785, 0.4829, 0.4780, 0.4769; %C3-DAN
               0.4929, 0.4774, 0.4719, 0.4773, 0.4790; %T3-DAN
               0.4690, 0.4498, 0.4466, 0.4598, 0.4582]; %O1-DAN
% mean_values = [0.6562, 0.6467, 0.6244, 0.6221, 0.6420; %F3-VIS
%                0.6301, 0.6565, 0.6633, 0.6571, 0.6847; %F4-VIS
%                0.6537, 0.6484, 0.6278, 0.6312, 0.6232; %C3-VIS
%                0.6296, 0.6524, 0.6329, 0.6302, 0.5792; %T3-VIS
%                0.6670, 0.6379, 0.6572, 0.6045, 0.6367]; %O1-VIS
% mean_values = [0.3955, 0.3938, 0.3914, 0.3929, 0.3840; %F3-left
%                0.4100, 0.4169, 0.4168, 0.4068, 0.4328; %F4-left
%                0.4042, 0.3990, 0.3972, 0.3988, 0.4026; %C3-left
%                0.3964, 0.3941, 0.3904, 0.3931, 0.3878; %T3-left
%                0.3811, 0.3779, 0.3721, 0.3746, 0.3767]; %O1-left
% mean_values = [0.4156, 0.4197, 0.4159, 0.4063, 0.4103; %F3-right
%                0.4091, 0.4078, 0.3990, 0.3986, 0.4006; %F4-right
%                0.4070, 0.4156, 0.4068, 0.4156, 0.4068; %C3-right
%                0.4016, 0.4119, 0.3979, 0.4098, 0.4123; %T3-right
%                0.4138, 0.4048, 0.4029, 0.3996, 0.4051]; %O1-right
% mean_values = [0.3814, 0.3842, 0.3860, 0.3753, 0.3758; % P-A               
%                0.3554, 0.3489, 0.3437, 0.3480, 0.3419; % A-P
%                0.3878, 0.3874, 0.3851, 0.3801, 0.3795; %Deymed 70BF
%                0.3794, 0.3818, 0.3800, 0.3801, 0.3826; %Deymed 50BF
%                0.3660, 0.3625, 0.3642, 0.3497, 0.3521; % Cool-B65
%                0.3567, 0.3562, 0.3513, 0.3529, 0.3503]; % DCC
% mean_values = [0.3955, 0.3938, 0.3914, 0.3929, 0.3840; %P-A left               
%                0.3760, 0.3673, 0.3666, 0.3677, 0.3675; % A-P
%                0.4043, 0.4038, 0.3984, 0.3908, 0.3926; %Deymed 70BF
%                0.4024, 0.3966, 0.4014, 0.3887, 0.3956; %Deymed 50BF
%                0.3947, 0.3809, 0.3919, 0.3741, 0.3742; % Cool-B65
%                0.3727, 0.3735, 0.3754, 0.3744, 0.3703]; % DCC
% mean_values = [0.4156, 0.4197, 0.4159, 0.4063, 0.4103; %P-A right             
%                0.3825, 0.3760, 0.3749, 0.3773, 0.3728; % A-P
%                0.4160, 0.4202, 0.4124, 0.4097, 0.4058; %Deymed 70BF
%                0.3987, 0.4097, 0.4023, 0.4130, 0.4123; %Deymed 50BF
%                0.3903, 0.3895, 0.3868, 0.3739, 0.3848; % Cool-B65
%                0.3825, 0.3858, 0.3731, 0.3794, 0.3787]; % DCC
% mean_values = [0.4189, 0.4421, 0.4328, 0.4083, 0.4282; %P-A DMN              
%                0.4120, 0.3966, 0.3901, 0.3959, 0.3922; % A-P
%                0.4491, 0.4367, 0.4462, 0.4298, 0.4232; %Deymed 70BF
%                0.4335, 0.4371, 0.4197, 0.4172, 0.4263; %Deymed 50BF
%                0.4336, 0.4157, 0.4117, 0.4073, 0.4084; % Cool-B65
%                0.3967, 0.3973, 0.3924, 0.3997, 0.3926]; % DCC
% mean_values = [0.4680, 0.4893, 0.4768, 0.4806, 0.4530; %P-A DAN              
%                0.4562, 0.4424, 0.4396, 0.4396, 0.4389; % A-P
%                0.4884, 0.5027, 0.4891, 0.4985, 0.4821; %Deymed 70BF
%                0.4870, 0.4819, 0.4775, 0.4842, 0.4825; %Deymed 50BF
%                0.4666, 0.4537, 0.4535, 0.4497, 0.4530; % Cool-B65
%                0.4461, 0.4475, 0.4530, 0.4397, 0.4434]; % DCC
% mean_values = [0.6562, 0.6467, 0.6244, 0.6221, 0.6420; %P-A VIS              
%                0.6357, 0.6053, 0.6207, 0.6180, 0.6148; % A-P
%                0.6687, 0.6335, 0.6309, 0.6168, 0.6310; %Deymed 70BF
%                0.6578, 0.6430, 0.6703, 0.6307, 0.6412; %Deymed 50BF
%                0.6364, 0.6471, 0.6809, 0.6392, 0.6113; % Cool-B65
%                0.6647, 0.6775, 0.6519, 0.6292, 0.6704]; % DCC
% 未刺激时的基线相关性
baseline_corr = 0.47155;
% 0.3967;%DMN 0.39796;%left 0.4071;%right 0.69241;%VIS 0.47155;%DAN
figure;
hold on;
set(gcf, 'Position', [1000,200,630,510], 'Renderer', 'painters');
% 生成颜色映射，确保不同刺激强度颜色一致
colors = lines(nStim);
legendHandles = gobjects(nStim, 1);

% 定义不同刺激强度的散点形状
markers = {'s', 's', 's', 's', 's'}; 

for coil = 1:nCoils
    for stim = 1:nStim
        % 突出标注低于或高于基线的情况（不同颜色）
        % if mean_values(coil, stim) < baseline_corr
        %     color = [1, 0, 0]; % 红色表示低于基线
        % else
            color = colors(stim, :); % 正常颜色
        % end
        
        % 确保同等强度下使用相同的散点形状
        h = scatter(coil, mean_values(coil, stim), 60, color, markers{stim}, 'filled');
        
        % 连线（若非第一列则与上一线圈同刺激强度连线）
        if coil > 1
            line([coil-1, coil], [mean_values(coil-1, stim), mean_values(coil, stim)], ...
                'Color', colors(stim, :), 'LineWidth', 2);
        end
        
        % 仅在第一次绘制时设置 legend
        if coil == 1
            legendHandles(stim) = h;
        end
    end
end


% 添加基线参考线
plot([0.5, nCoils + 0.5], [baseline_corr, baseline_corr], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);

% 在虚线右侧标注"Without TMS"
text(nCoils + 0.05, baseline_corr, 'Without TMS', 'Color', [0.5 0.5 0.5], ...
    'FontSize', ticklabelsize, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

% 设置横坐标
% set(gca, 'XTick', 1:nCoils, 'XTickLabel',[]);
set(gca, 'XTick', 1:nCoils, 'XTickLabel',coil_names,...
    'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');

% 使横坐标不贴边界
xlim([0.5, nCoils + 0.5]);
ylim([0.44 0.51]);
yticks(0.44:0.01:0.51);
% 设置纵坐标标签（加斜体）
ylabel('\it\gamma_{DAN}', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
% 分别设置 X/Y 轴字体属性
ax = gca;
ax.XAxis.FontSize = ticklabelsize;  % X 轴字体稍小
ax.YAxis.FontSize = ticklabelsize;  % Y 轴字体保持原始大小
ax.XAxis.FontName = fontname;
ax.YAxis.FontName = fontname;
ax.XAxis.FontWeight = 'bold';
ax.YAxis.FontWeight = 'bold';
% 调整刻度标签（加粗）
% set(gca, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
% title('','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
% % 调整图例
% legend(legendHandles, arrayfun(@(x) [num2str(x) '% MT'], stim_range, 'UniformOutput', false), ...
%     'FontSize', 10, 'Location', 'northeast', 'Box', 'off');
% 
% % 调整 legend 位置（右移但不超出图外）
% legendPosition = get(legend, 'Position');
% legendPosition(1) = legendPosition(1) + 0.05; % 右移
% set(legend, 'Position', legendPosition);
% 
% % 设置 legend 背景透明
% legend('Color', 'none');

hold off;

%% 基线电位随全局强度参数变化曲线
figure % 基线电位随全局耦合参数变化曲线
load('Data\baseline_potential.mat')
fontsize=12;
ticklabelsize = 10;
fontname='Times New Roman';

% 设定横坐标（强度范围 80%MT - 120%MT）
x_values = linspace(80, 120, 41); % x 轴从 80 到 120
y_values = 1:68; % 脑区索引

% 统一 colormap
colormap("parula");

% ======= 主图（80%-120%MT范围） =======
subplot('Position', [0.1, 0.15, 0.6, 0.7]) % 左侧大图，占 60% 宽度
imagesc(x_values, y_values, baseline_potential', [1 9]) % baseline_potential' 转置
set(gca, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold')
xlabel('Strength (%MT)', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold')
ylabel('Region', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold')
title('Baseline potential (mV)','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
clim([1 9]); % 统一颜色范围

% ======= 颜色条（基线电位） =======
baseline_values = [... % 68个脑区的基线电位
                   6.83, 1.78, 7.13, 6.90, 1.07, 7.04, 7.69, 7.01, 6.55, 7.74, ...
                   6.88, 7.14, 2.46, 7.19, 1.65, 6.97, 6.87, 1.63, 6.78, 6.89, ...
                   7.52, 6.79, 8.01, 7.42, 1.54, 7.85, 8.53, 7.95, 7.32, 7.54, ...
                   0.99, 1.28, 1.51, 7.06, 6.75, 2.10, 7.15, 6.91, 1.03, 6.97, ...
                   7.99, 6.99, 6.04, 7.63, 6.77, 7.11, 3.92, 7.27, 1.71, 7.03, ...
                   6.88, 1.86, 6.85, 6.92, 7.59, 6.83, 7.98, 7.47, 1.50, 7.82, ...
                   8.37, 7.98, 7.29, 7.47, 1.07, 1.23, 1.33, 7.13];
baseline_values_200 = [...% 68个脑区的前200ms基线电位
                   6.74, 1.76, 7.16, 6.67, 1.07, 7.10, 7.71, 6.96, 6.09, 7.71, ...
                   6.87, 7.10, 2.22, 7.16, 1.63, 6.94, 6.77, 1.60, 6.51, 6.62, ...
                   7.57, 6.82, 7.99, 7.45, 1.52, 7.88, 8.41, 7.94, 7.29, 7.59, ...
                   0.98, 1.32, 1.51, 7.07, 6.72, 2.04, 7.13, 6.85, 1.04, 7.00, ...
                   7.96, 6.89, 4.75, 7.58, 6.72, 7.12, 2.56, 7.23, 1.64, 6.92, ...
                   6.81, 1.83, 6.82, 6.88, 7.57, 6.53, 7.96, 7.48, 1.43, 7.77, ...
                   8.27, 7.98, 7.27, 7.55, 1.06, 1.22, 1.31, 7.05];
baseline_values_800 = [...% 68个脑区的后200ms基线电位
                   6.93, 1.80, 7.18, 6.98, 1.05, 7.03, 7.73, 7.04, 6.90, 7.74, ...
                   6.94, 7.14, 2.58, 7.18, 1.67, 7.00, 6.97, 1.63, 6.88, 6.96, ...
                   7.59, 6.93, 8.08, 7.45, 1.54, 7.94, 8.56, 8.00, 7.31, 7.56, ...
                   0.98, 1.29, 1.52, 7.08, 6.92, 2.21, 7.13, 6.97, 1.02, 6.99, ...
                   8.05, 6.86, 6.70, 7.61, 6.97, 7.15, 4.90, 7.30, 1.76, 7.05, ...
                   6.93, 1.90, 6.97, 6.98, 7.67, 6.93, 8.06, 7.52, 1.52, 7.95, ...
                   8.51, 7.99, 7.30, 7.57, 1.09, 1.22, 1.34, 7.16];
subplot('Position', [0.72, 0.15, 0.05, 0.7]) % 颜色条位于中间，占 5% 宽度
imagesc([0, 1], y_values, baseline_values', [1 9]); % 数据 1x68 形式
clim([1 9]); % 颜色范围与主图一致
axis off; % 隐藏坐标轴（只显示颜色映射）
xlim([0, 1]); % 让颜色条紧凑
title('No TMS','FontSize',8, 'FontWeight', 'bold') % 标题

% ======= 右侧 colorbar（对应主图） =======
subplot('Position', [0.80, 0.15, 0.04, 0.7]) % 右侧 colorbar，占 4% 宽度
c = colorbar; % 右侧 colorbar
c.Position = [0.80, 0.15, 0.04, 0.7]; % 调整 colorbar 位置
clim([1 9]); % 统一 colorbar 颜色范围
c.Ticks = 1:9; % 设置 colorbar 刻度
set(gca, 'Visible', 'off', 'FontWeight', 'bold') % 隐藏 colorbar 的背景坐标轴

% ======= 保存 =======
saveas(gcf, 'Figure\baseline_potential.fig')

%% 主频随全局强度参数变化曲线
figure % 主频随全局耦合参数变化曲线
load('Data\dominant_rhythm.mat')
fontsize=12;
ticklabelsize = 10;
fontname='Times New Roman';

% 设定横坐标（强度范围 80%MT - 120%MT）
x_values = linspace(80, 120, 41); % x 轴从 80 到 120
y_values = 1:68; % 脑区索引

% 统一 colormap
colormap("parula");

% ======= 主图（80%-120%MT范围） =======
subplot('Position', [0.1, 0.15, 0.6, 0.7]) % 左侧大图，占 60% 宽度
imagesc(x_values, y_values, dominant_rhythm', [4 11]) % baseline_potential' 转置
set(gca, 'FontSize', ticklabelsize, 'FontName', fontname,'FontWeight', 'bold')
xlabel('Strength (%MT)', 'FontSize', fontsize, 'FontName', fontname,'FontWeight', 'bold')
ylabel('Region', 'FontSize', fontsize, 'FontName', fontname,'FontWeight', 'bold')
title('Dominant rhythm (Hz)','FontSize',fontsize,'FontName',fontname,'FontWeight', 'bold')
clim([4 11]); % 统一颜色范围

% ======= 颜色条（主频） =======
dominant_values = [... % 68个脑区的主频
                   9.45312500000000, 6.19140625000000, 8.82812500000000, 9.43359375000000, 5.03906250000000, 8.78906250000000, 9.02343750000000, 8.86718750000000, 8.94531250000000, 9.06250000000000,...
                   9.70703125000000, 9.19921875000000, 3.80859375000000, 9.27734375000000, 4.45312500000000, 9.62890625000000, 8.90625000000000, 5.74218750000000, 8.90625000000000, 9.41406250000000,...
                   8.90625000000000, 9.49218750000000, 8.82812500000000, 9.29687500000000, 6.11328125000000, 8.73046875000000, 8.80859375000000, 9.06250000000000, 9.27734375000000, 9.21875000000000,...
                   4.14062500000000, 5.25390625000000, 5.52734375000000, 9.76562500000000, 8.92578125000000, 4.66796875000000, 9.04296875000000, 9.27734375000000, 4.84375000000000, 8.84765625000000,...
                   8.84765625000000, 8.82812500000000, 7.10937500000000, 8.88671875000000, 9.08203125000000, 9.21875000000000, 4.84375000000000, 9.04296875000000, 4.94140625000000, 9.55078125000000,...
                   9.02343750000000, 5.68359375000000, 9.04296875000000, 9.41406250000000, 8.88671875000000, 9.45312500000000, 8.94531250000000, 9.16015625000000, 5.33203125000000, 8.73046875000000,...
                   8.76953125000000, 8.88671875000000, 8.94531250000000, 9.02343750000000, 5.01953125000000, 4.70703125000000, 5.19531250000000, 9.58984375000000];
dominant_values_200 = [...
                9.61, 8.59, 9.45, 10.16, 10.63, 8.52, 8.83, 8.75, 8.59, 8.91, 10.00, 9.69, 8.36, 8.91, 10.08, 9.77, 8.59, 8.83, 9.06, 9.45, ...
                9.14, 10.47, 8.98, 9.61, 9.30, 8.83, 8.36, 8.67, 8.75, 9.14, 10.39, 9.77, 9.06, 9.92, 9.45, 9.22, 8.98, 9.61, 10.23, 8.52, ...
                8.05, 8.59, 8.52, 8.67, 10.16, 9.38, 8.98, 8.67, 8.59, 9.69, 8.67, 8.91, 8.91, 9.22, 8.91, 9.77, 8.44, 8.59, 9.45, 8.52, ...
                8.20, 8.13, 8.91, 8.36, 10.63, 9.84, 10.94, 9.53];
dominant_values_800 = [...
                10.16, 9.22, 8.91, 9.77, 11.09, 9.14, 8.91, 8.67, 10.70, 8.83, 10.78, 9.22, 9.92, 9.06, 10.31, 10.23, 8.75, 9.22, 9.14, 9.69, ...
                9.06, 10.78, 8.52, 9.61, 10.31, 8.59, 8.20, 8.75, 9.30, 9.14, 10.39, 10.00, 10.08, 9.84, 9.30, 8.75, 8.91, 9.77, 10.47, 8.75, ...
                8.59, 9.14, 10.16, 8.83, 10.86, 9.61, 8.83, 8.28, 9.61, 9.92, 9.14, 9.53, 8.75, 9.92, 8.83, 10.78, 8.75, 9.06, 9.69, 8.28, ...
                8.20, 8.67, 8.83, 8.59, 10.39, 11.09, 9.38, 9.92];
subplot('Position', [0.72, 0.15, 0.05, 0.7]) % 颜色条位于中间，占 5% 宽度
imagesc([0, 1], y_values, dominant_values', [4 11]); % 数据 1x68 形式
clim([4 11]); % 颜色范围与主图一致
axis off; % 隐藏坐标轴（只显示颜色映射）
xlim([0, 1]); % 让颜色条紧凑
title('No TMS','FontSize',8, 'FontWeight', 'bold') % 标题

% ======= 右侧 colorbar（对应主图） =======
subplot('Position', [0.80, 0.15, 0.04, 0.7]) % 右侧 colorbar，占 4% 宽度
c = colorbar; % 右侧 colorbar
c.Position = [0.80, 0.15, 0.04, 0.7]; % 调整 colorbar 位置
clim([4 11]); % 统一 colorbar 颜色范围
c.Ticks = 4:11; % 设置 colorbar 刻度
set(gca, 'Visible', 'off','FontWeight', 'bold') % 隐藏 colorbar 的背景坐标轴

% ======= 保存 =======
saveas(gcf, 'Figure\dominant_rhythm.fig')
%% 香农熵随全局强度参数变化曲线
figure % 香农熵随全局强度参数变化曲线
load('Data\pe_values.mat')
fontsize=12;
ticklabelsize = 10;
fontname='Times New Roman';

% 设定横坐标（强度范围 80%MT - 120%MT）
x_values = linspace(80, 120, 41); % x 轴从 80 到 120
y_values = 1:68; % 脑区索引

% 统一 colormap
colormap("parula");

% ======= 主图（80%-120%MT范围） =======
subplot('Position', [0.1, 0.15, 0.6, 0.7]) % 左侧大图，占 60% 宽度
imagesc(x_values, y_values, pe_values', [2.5 4]) % baseline_potential' 转置
set(gca, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold')
xlabel('Strength (%MT)', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold')
ylabel('Region', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold')
title('Shannon entropy','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
clim([2.5 4]); % 统一颜色范围

% ======= 颜色条（排序熵） =======
Shannon_entropy = [... % 68个脑区的
    3.44645348779322  3.60154631651915  3.49091252594292  3.53984357506273 ...
    3.64381444360979  3.36736937587834  3.45220657450495  3.40605622214912 ...
    3.45615126918319  3.45673577089172  3.47979444131962  3.46845342175577 ...
    3.46259863048833  3.34587024811229  3.61097473565773  3.45685574709025 ...
    3.44115095409301  3.76035469736117  3.51563904832952  3.52843209474492 ...
    3.41640190368217  3.50027376529512  3.44684512831702  3.50226645981174 ...
    3.75411210886757  3.49370922154466  3.45111495803956  3.39945813131641 ...
    3.46028199590728  3.43175700888431  3.64087610223711  3.68783961914349 ...
    3.70859845866598  3.52967394015773  3.48728080465800  3.51713443469037 ...
    3.47528612665919  3.48183943129233  3.59823411476552  3.34368892647509 ...
    3.42857649489736  3.40125401537694  3.33265908006786  3.53522613057179 ...
    3.46990213876241  3.47708323701040  3.05382571443198  3.36064926215338 ...
    3.62147850041958  3.53724455444703  3.51886680058337  3.57600427832707 ...
    3.50782069667835  3.47436518499777  3.41243632330692  3.52231630115383 ...
    3.48669357453406  3.48585255125603  3.72114977261582  3.40051105528888 ...
    3.41967168986875  3.41928036423999  3.42896715180753  3.32878572648330 ...
    3.63354683227226  3.66127574594639  3.68291525178161  3.49678930758026];
Shannon_entropy_200 = [... % 68个脑区的
    2.61  2.65  2.68  2.62  2.62  2.57  2.52  2.58 ...
    2.51  2.60  2.60  2.60  2.65  2.67  2.67  2.61 ...
    2.65  2.68  2.49  2.62  2.55  2.58  2.63  2.70 ...
    2.66  2.55  2.55  2.51  2.61  2.62  2.68  2.62 ...
    2.67  2.59  2.54  2.73  2.65  2.55  2.60  2.56 ...
    2.53  2.65  2.65  2.63  2.55  2.60  2.66  2.54 ...
    2.68  2.61  2.58  2.69  2.57  2.60  2.60  2.62 ...
    2.56  2.59  2.73  2.51  2.50  2.55  2.61  2.48 ...
    2.71  2.66  2.67  2.59];
Shannon_entropy_800 = [...
    2.63 2.76 2.59 2.61 2.69 2.57 2.50 2.62 2.64 2.53 ...
    2.58 2.63 2.68 2.63 2.72 2.64 2.64 2.63 2.59 2.62 ...
    2.60 2.69 2.57 2.64 2.67 2.49 2.53 2.60 2.63 2.61 ...
    2.73 2.70 2.61 2.59 2.64 2.65 2.64 2.62 2.62 2.50 ...
    2.61 2.62 2.57 2.53 2.63 2.54 2.56 2.64 2.64 2.59 ...
    2.60 2.66 2.54 2.62 2.56 2.68 2.59 2.59 2.61 2.55 ...
    2.49 2.62 2.58 2.53 2.71 2.65 2.68 2.56];
subplot('Position', [0.72, 0.15, 0.05, 0.7]) % 颜色条位于中间，占 5% 宽度
imagesc([0, 1], y_values, Shannon_entropy', [2.5 4]); % 数据 1x68 形式
clim([2.5 4]); % 颜色范围与主图一致
axis off; % 隐藏坐标轴（只显示颜色映射）
xlim([0, 1]); % 让颜色条紧凑
title('No TMS','FontSize',8, 'FontWeight', 'bold') % 标题

% ======= 右侧 colorbar（对应主图） =======
subplot('Position', [0.80, 0.15, 0.04, 0.7]) % 右侧 colorbar，占 4% 宽度
c = colorbar; % 右侧 colorbar
c.Position = [0.80, 0.15, 0.04, 0.7]; % 调整 colorbar 位置
clim([2.5 4]); % 统一 colorbar 颜色范围
c.Ticks = 2.5:0.3:4; % 设置 colorbar 刻度
set(gca, 'Visible', 'off', 'FontWeight', 'bold') % 隐藏 colorbar 的背景坐标轴

% ======= 保存 =======
saveas(gcf, 'Figure\baseline_potential.fig')
%% Morlet 小波变换
load("Data/excs.mat"); % 假设数据已存储在 excs 变量中
% 参数设定
fs = 1000; % 采样频率 (Hz)
timeWindow = 501:2000; % 选定时间窗口 (刺激后 0-500ms)
regionIdx = 26; % 指定脑区索引

% 提取目标时间段的信号
dataSegment = excs(timeWindow, regionIdx);
dataSegment = dataSegment - mean(dataSegment); % 去均值防止漂移

% 设定小波参数
frequencies = 0:40; % 关注 5Hz 到 30Hz 之间的频率
wavenumber = 6; % Morlet 小波的波数参数（控制时间-频率分辨率）
% 计算 Morlet 小波变换
[wt, f] = cwt(dataSegment, fs, 'amor', 'FrequencyLimits', [frequencies(1), frequencies(end)], 'VoicesPerOctave', 32);

% 可视化小波变换结果
time = (timeWindow - 1000); % 转换为相对刺激时间 (ms)
figure;
set(gcf, 'Position', [300,100,550,400]);
imagesc(time, f, abs(wt)); % 取幅值
axis xy;
set(gca,'xlim',[-500 1000])
set(gca,'xtick',-500:250:1000)
% xlabel('时间 (毫秒)');
% ylabel('频率 (Hz)');
% title(['脑区 ', num2str(regionIdx), ' 的 Morlet 小波变换']);
xlabel('time (ms)');
ylabel('Frequency (Hz)');
title(['Region ', num2str(regionIdx)]);
colorbar;
clim([0 max(abs(wt(:)))]) % 颜色归一化
%% STFT 替换小波变换
load("Data/excs.mat"); % 假设数据已存储在 excs 变量中
% 参数设定
fs = 1000; % 采样频率 (Hz)
timeWindow = 501:2000; % 选定时间窗口 (刺激后 0-500ms)
regionIdx = 30; % 指定脑区索引

% 提取目标时间段的信号
dataSegment = excs(timeWindow, regionIdx);
dataSegment = dataSegment - mean(dataSegment); % 去均值防止漂移

% STFT 参数设定
windowLength = 256; % 窗口长度
overlap = 240; % 重叠点数（增加重叠以提高时间分辨率）
nfft = 1024; % 增加 FFT 点数以提高频率分辨率
frequencies = 0:41; % 关注 0Hz 到 40Hz 之间的频率

% 计算 STFT
[s, f, t] = spectrogram(dataSegment, windowLength, overlap, nfft, fs, 'yaxis');

% 限制频率范围在 0-40Hz
freqRange = (f >= frequencies(1)) & (f <= frequencies(end)); % 选择 0-40Hz 的频率范围
s = s(freqRange, :); % 仅保留 0-40Hz 的数据
f = f(freqRange); % 更新频率轴

% 插值以提高图像平滑度
time = (timeWindow - 1000); % 转换为相对刺激时间 (ms)
timeSTFT = linspace(time(1), time(end), size(s, 2)); % 时间轴

% 创建网格
[T_grid, F_grid] = meshgrid(timeSTFT, f); % 网格用于插值
S_abs = abs(s); % 取 STFT 的幅值

% 插值
[T_interp, F_interp] = meshgrid(linspace(time(1), time(end), 500), linspace(f(1), f(end), 500)); % 更密集的网格
S_interp = griddata(T_grid(:), F_grid(:), S_abs(:), T_interp, F_interp, 'cubic'); % 插值

% 可视化 STFT 结果
figure;
set(gcf, 'Position', [300,100,550,400]);
imagesc(T_interp(1,:), F_interp(:,1), S_interp); % 绘制插值后的结果
axis xy;
set(gca,'xlim',[-500 1000])
set(gca,'ylim',[frequencies(1) frequencies(end)-1])
set(gca,'xtick',-500:250:1000)
set(gca,'ytick',0:5:frequencies(end)-1)
xlabel('time (ms)');
ylabel('Frequency (Hz)');
title(['Region ', num2str(regionIdx), ' 的 STFT']);
colorbar;
clim([0 max(S_abs(:))]) % 颜色归一化
%% SST 替换 STFT（优化）
load("Data/excs.mat"); % 假设数据已存储在 excs 变量中
% 参数设定
fs = 1000; % 采样频率 (Hz)
timeWindow = 501:2000; % 选定时间窗口 (刺激后 0-500ms)
regionIdx = 30; % 指定脑区索引

% 提取目标时间段的信号
dataSegment = excs(timeWindow, regionIdx);
dataSegment = dataSegment - mean(dataSegment); % 去均值防止漂移

% SST 参数设定（增加频率分辨率）
frequencies = 1:41; % 关注 0Hz 到 40Hz 之间的频率
voicePerOctave = 48; % 提高每倍频程的频率数

% 计算 SST
[ss, f] = wsst(dataSegment, fs, 'VoicesPerOctave', voicePerOctave);

% 限制频率范围在 0-40Hz
freqRange = (f >= frequencies(1)) & (f <= frequencies(end));
ss = ss(freqRange, :);
f = f(freqRange);

% 时间轴
time = (timeWindow - 1000);
timeSST = linspace(time(1), time(end), size(ss, 2)); 

% ⚠️ 使用插值增强显示效果
[X, Y] = meshgrid(timeSST, f);
[Xq, Yq] = meshgrid(linspace(min(timeSST), max(timeSST), 1000), linspace(min(f), max(f), 400));
ss_interp = interp2(X, Y, abs(ss), Xq, Yq, 'cubic');

% 可视化
figure;
set(gcf, 'Position', [300,100,540,400], 'Renderer', 'painters');

% 使用平滑的插值结果
imagesc(Xq(1, :), Yq(:, 1), ss_interp);

axis xy;
set(gca, 'xlim', [-500 1000]);
set(gca, 'xtick', -500:250:1000);
xlabel('Time (ms)','FontSize',fontsize,'FontName',fontname,'FontWeight', 'bold');
ylabel('Frequency (Hz)','FontSize',fontsize,'FontName',fontname,'FontWeight', 'bold');
set(gca,'ylim',[frequencies(1) frequencies(end)])
set(gca,'ytick',0:5:frequencies(end))
title(['Region ', num2str(regionIdx)],'FontSize',fontsize,'FontName',fontname,'FontWeight', 'bold');
colormap('turbo'); % 使用平滑的颜色映射
colorbar;
set(gca,'FontWeight', 'bold')
% 归一化颜色映射
clim([0 max(ss_interp(:))]);
%% SST 整体
load("Data/excs.mat"); % excs: 3001 × 68
fs = 1000; % 采样频率
timeWindow = 501:2000;
frequencies = 1:41;
voicePerOctave = 48;

% 图形设置
fontname = 'Times New Roman';
fontsize = 8;

figure('Position', [100, 50, 1600, 2400], 'Renderer', 'painters');
tiledlayout(8, 9, 'Padding', 'compact', 'TileSpacing', 'compact');

for regionIdx = 1:68
    dataSegment = excs(timeWindow, regionIdx);
    dataSegment = dataSegment - mean(dataSegment);

    [ss, f] = wsst(dataSegment, fs, 'VoicesPerOctave', voicePerOctave);
    freqRange = (f >= frequencies(1)) & (f <= frequencies(end));
    ss = ss(freqRange, :);
    f = f(freqRange);
    
    % 时间轴
    time = (timeWindow - 1000);
    timeSST = linspace(time(1), time(end), size(ss, 2)); 
    
    % 插值增强
    [X, Y] = meshgrid(timeSST, f);
    [Xq, Yq] = meshgrid(linspace(min(timeSST), max(timeSST), 1000), linspace(min(f), max(f), 400));
    ss_interp = interp2(X, Y, abs(ss), Xq, Yq, 'spline');

    % 绘图
    nexttile;
    imagesc(Xq(1,:), Yq(:,1), ss_interp);
    axis xy;
    colormap('turbo');
    clim([0, max(ss_interp(:))]); % 每个子图使用独立颜色上限
    
    xlim([-500 1000]);
    ylim([frequencies(1) frequencies(end)]);

    set(gca, ...
        'FontName', fontname, ...
        'FontSize', fontsize, ...
        'XTick', -500:500:1000, ...
        'YTick', 0:10:frequencies(end));

    title(['Region ', num2str(regionIdx)], ...
        'FontSize', fontsize, ...
        'FontName', fontname, ...
        'FontWeight', 'bold');

    if mod(regionIdx-1, 9) == 0
        ylabel('Frequency (Hz)', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
    else
        yticks([]);
    end

    xlabel('Time (ms)', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
end
%% SST-TFP（优化）
load("Data/excs.mat"); % 假设数据已存储在 excs 变量中
% 参数设定
fs = 1000; % 采样频率 (Hz)
timeWindow = 501:2000; % 选定时间窗口 (刺激后 0-500ms)
regionIdx = 1; % 指定脑区索引

% 提取目标时间段的信号
dataSegment = excs(timeWindow, regionIdx);
dataSegment = dataSegment - mean(dataSegment); % 去均值防止漂移

% SST 参数设定（提高频率分辨率）
frequencies = 1:41; % 关注 0Hz 到 40Hz 之间的频率
voicePerOctave = 48; % 最大为 48

% 计算 SST-TFP
[ss, f] = wsst(dataSegment, fs, 'VoicesPerOctave', voicePerOctave);

% 限制频率范围在 0-40Hz
freqRange = (f >= frequencies(1)) & (f <= frequencies(end));
ss = ss(freqRange, :);
f = f(freqRange);

% 通过相位重定位进行时间-频率重构
tfp = abs(ss).^2;

% 时间轴
time = (timeWindow - 1000);
timeSST = linspace(time(1), time(end), size(tfp, 2)); 

% ⚠️ 使用插值增强显示效果
[X, Y] = meshgrid(timeSST, f);
[Xq, Yq] = meshgrid(linspace(min(timeSST), max(timeSST), 500), linspace(min(f), max(f), 200));
tfp_interp = interp2(X, Y, tfp, Xq, Yq, 'cubic');

% 可视化
figure;
set(gcf, 'Position', [300,100,600,400]);

% 使用平滑的插值结果
imagesc(Xq(1, :), Yq(:, 1), tfp_interp);

axis xy;
set(gca, 'xlim', [-500 1000]);
set(gca, 'xtick', -500:250:1000);
set(gca,'ylim',[frequencies(1) frequencies(end)])
set(gca,'ytick',0:5:frequencies(end))
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(['Region ', num2str(regionIdx), ' 的 SST-TFP']);
colormap('turbo'); % 使用平滑的颜色映射
colorbar;
ylabel(colorbar, 'Power (mV^2)');

% 归一化颜色映射
clim([0 max(tfp_interp(:))]);
%% 图论分析
load('Data\Cij.mat')
load('Data\sim_plv_80.mat')
load('Data\sim_plv_90.mat')
load('Data\sim_plv_100.mat')
load('Data\sim_plv_110.mat')
load('Data\sim_plv_120.mat')
load('Data\sim_plv_origin.mat')
sim_degree_80=[];sim_Eglo_80=[];sim_cpl_80=[];sim_cc_80=[];
sim_degree_90=[];sim_Eglo_90=[];sim_cpl_90=[];sim_cc_90=[];
sim_degree_100=[];sim_Eglo_100=[];sim_cpl_100=[];sim_cc_100=[];
sim_degree_110=[];sim_Eglo_110=[];sim_cpl_110=[];sim_cc_110=[];
sim_degree_120=[];sim_Eglo_120=[];sim_cpl_120=[];sim_cc_120=[];
sim_degree_origin=[];sim_Eglo_origin=[];sim_cpl_origin=[];sim_cc_origin=[];
sim_degree_rand_80=[];sim_Eglo_rand_80=[];sim_cpl_rand_80=[];sim_cc_rand_80=[];
sim_degree_rand_90=[];sim_Eglo_rand_90=[];sim_cpl_rand_90=[];sim_cc_rand_90=[];
sim_degree_rand_100=[];sim_Eglo_rand_100=[];sim_cpl_rand_100=[];sim_cc_rand_100=[];
sim_degree_rand_110=[];sim_Eglo_rand_110=[];sim_cpl_rand_110=[];sim_cc_rand_110=[];
sim_degree_rand_120=[];sim_Eglo_rand_120=[];sim_cpl_rand_120=[];sim_cc_rand_120=[];
sim_degree_origin_rand=[];sim_Eglo_origin_rand=[];sim_cpl_origin_rand=[];sim_cc_origin_rand=[];
sim_plv_nodenum_80=[];sim_plv_nodenum_90=[];sim_plv_nodenum_100=[];sim_plv_nodenum_110=[];sim_plv_nodenum_120=[];sim_plv_origin_nodenum=[];
strength = strengths_und(Cij);
threshold = 0:0.01:0.5;
n = 1;
for i = threshold
    sim_plv_thre_80 = threshold_absolute(sim_plv_80,i);
    sim_plv_thre_90 = threshold_absolute(sim_plv_90,i);
    sim_plv_thre_100 = threshold_absolute(sim_plv_100,i);
    sim_plv_thre_110 = threshold_absolute(sim_plv_110,i);
    sim_plv_thre_120 = threshold_absolute(sim_plv_120,i);
    sim_plv_origin_thre = threshold_absolute(sim_plv_origin,i);
    [sim_degree_80(n),sim_Eglo_80(n),sim_cpl_80(n),sim_cc_80(n)] = anal_cm(sim_plv_thre_80);
    [sim_degree_90(n),sim_Eglo_90(n),sim_cpl_90(n),sim_cc_90(n)] = anal_cm(sim_plv_thre_90);
    [sim_degree_100(n),sim_Eglo_100(n),sim_cpl_100(n),sim_cc_100(n)] = anal_cm(sim_plv_thre_100);
    [sim_degree_110(n),sim_Eglo_110(n),sim_cpl_110(n),sim_cc_110(n)] = anal_cm(sim_plv_thre_110);
    [sim_degree_120(n),sim_Eglo_120(n),sim_cpl_120(n),sim_cc_120(n)] = anal_cm(sim_plv_thre_120);
    [sim_degree_origin(n),sim_Eglo_origin(n),sim_cpl_origin(n),sim_cc_origin(n)] = anal_cm(sim_plv_origin_thre);
    sim_plv_thre_rand_80=makerandCIJ_und(size(sim_plv_thre_80,1),size(sim_plv_thre_80(sim_plv_thre_80~=0),1)/2);
    sim_plv_thre_rand_90=makerandCIJ_und(size(sim_plv_thre_90,1),size(sim_plv_thre_90(sim_plv_thre_90~=0),1)/2);
    sim_plv_thre_rand_100=makerandCIJ_und(size(sim_plv_thre_100,1),size(sim_plv_thre_100(sim_plv_thre_100~=0),1)/2);
    sim_plv_thre_rand_110=makerandCIJ_und(size(sim_plv_thre_110,1),size(sim_plv_thre_110(sim_plv_thre_110~=0),1)/2);
    sim_plv_thre_rand_120=makerandCIJ_und(size(sim_plv_thre_120,1),size(sim_plv_thre_120(sim_plv_thre_120~=0),1)/2);
    sim_plv_origin_thre_rand=makerandCIJ_und(size(sim_plv_origin_thre,1),size(sim_plv_origin_thre(sim_plv_origin_thre~=0),1)/2);
    sim_plv_thre_rand_80=randmio_und(sim_plv_thre_rand_80,10);
    sim_plv_thre_rand_90=randmio_und(sim_plv_thre_rand_90,10);
    sim_plv_thre_rand_100=randmio_und(sim_plv_thre_rand_100,10);
    sim_plv_thre_rand_110=randmio_und(sim_plv_thre_rand_110,10);
    sim_plv_thre_rand_120=randmio_und(sim_plv_thre_rand_120,10);
    sim_plv_origin_thre_rand=randmio_und(sim_plv_origin_thre_rand,10);
    [~,~,sim_cpl_rand_80(n),sim_cc_rand_80(n)] = anal_cm(sim_plv_thre_rand_80);
    [~,~,sim_cpl_rand_90(n),sim_cc_rand_90(n)] = anal_cm(sim_plv_thre_rand_90);
    [~,~,sim_cpl_rand_100(n),sim_cc_rand_100(n)] = anal_cm(sim_plv_thre_rand_100);
    [~,~,sim_cpl_rand_110(n),sim_cc_rand_110(n)] = anal_cm(sim_plv_thre_rand_110);
    [~,~,sim_cpl_rand_120(n),sim_cc_rand_120(n)] = anal_cm(sim_plv_thre_rand_120);
    [~,~,sim_cpl_origin_rand(n),sim_cc_origin_rand(n)] = anal_cm(sim_plv_origin_thre_rand);
    sim_plv_nodenum_80(n) = length(find(sum(sim_plv_thre_80)));
    sim_plv_nodenum_90(n) = length(find(sum(sim_plv_thre_90)));
    sim_plv_nodenum_100(n) = length(find(sum(sim_plv_thre_100)));
    sim_plv_nodenum_110(n) = length(find(sum(sim_plv_thre_110)));
    sim_plv_nodenum_120(n) = length(find(sum(sim_plv_thre_120)));
    sim_plv_origin_nodenum(n) = length(find(sum(sim_plv_origin_thre)));
    n = n+1;
end
sim_threshold_80 = Get_threhold(sim_plv_80);
sim_threshold_90 = Get_threhold(sim_plv_90);
sim_threshold_100 = Get_threhold(sim_plv_100);
sim_threshold_110 = Get_threhold(sim_plv_110);
sim_threshold_120 = Get_threhold(sim_plv_120);
sim_origin_threshold = Get_threhold(sim_plv_origin);
a = sim_cc_80./sim_cc_rand_80;b = sim_cpl_80./sim_cpl_rand_80;
c = sim_cc_90./sim_cc_rand_90;d = sim_cpl_90./sim_cpl_rand_90;
e = sim_cc_100./sim_cc_rand_100;f = sim_cpl_90./sim_cpl_rand_100;
g = sim_cc_110./sim_cc_rand_110;h = sim_cpl_90./sim_cpl_rand_110;
i = sim_cc_120./sim_cc_rand_120;j = sim_cpl_90./sim_cpl_rand_120;
k = sim_cc_origin./sim_cc_origin_rand;l = sim_cpl_origin./sim_cpl_origin_rand;
save('Data\graph_theoretical_plv_zitu.mat','threshold','strength',...
    'sim_degree_80','sim_Eglo_80','sim_cpl_80','sim_cc_80',...
    'sim_degree_90','sim_Eglo_90','sim_cpl_90','sim_cc_90',...
    'sim_degree_100','sim_Eglo_100','sim_cpl_100','sim_cc_100',...
    'sim_degree_110','sim_Eglo_110','sim_cpl_110','sim_cc_110',...
    'sim_degree_120','sim_Eglo_120','sim_cpl_120','sim_cc_120',...
    'sim_degree_origin', 'sim_Eglo_origin', 'sim_cpl_origin','sim_cc_origin', ...
    'sim_threshold_80','sim_threshold_90','sim_threshold_100','sim_threshold_110','sim_threshold_120','sim_origin_threshold',...
    'sim_plv_nodenum_80','sim_plv_nodenum_90','sim_plv_nodenum_100','sim_plv_nodenum_110','sim_plv_nodenum_120','sim_plv_origin_nodenum',...
    'sim_cpl_rand_80','sim_cc_rand_80','sim_cpl_rand_90','sim_cc_rand_90','sim_cpl_rand_100','sim_cc_rand_100','sim_cpl_rand_110','sim_cc_rand_110','sim_cpl_rand_120','sim_cc_rand_120', ...
    'sim_cpl_origin_rand','sim_cc_origin_rand')
%% 图论(五条)
% 定义字体参数（确保在运行前定义）
clc;clear;close all;
fontname = 'Times New Roman';
fontsize = 12;
ticklabelsize = 10;
% 节点数
figure 
load('Data\graph_theoretical_plv_zitu.mat')
hold on;
plot(threshold, sim_plv_origin_nodenum, 'b-o', 'LineWidth', 1);
plot(threshold, sim_plv_nodenum_80, 'm--s', 'LineWidth', 1);
plot(threshold, sim_plv_nodenum_90, 'c-.^', 'LineWidth', 1);
plot(threshold, sim_plv_nodenum_100, 'k-x', 'LineWidth', 1);
plot(threshold, sim_plv_nodenum_110, 'r--d', 'LineWidth', 1);
plot(threshold, sim_plv_nodenum_120, 'g-.p', 'LineWidth', 1);
hold off;
set(gca,'xlim',[0 0.5], 'xtick',0:0.1:0.5, 'ylim',[50 70], 'ytick',50:2:70,...
        'FontSize',ticklabelsize,'FontName',fontname, 'FontWeight', 'bold')
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Node number','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
% legend({'Without TMS','80% MT','90% MT','100% MT','110% MT','120% MT'},...
%        'FontSize',ticklabelsize,'FontName',fontname, 'Location','best')
% legend('boxoff')
saveas(gcf, 'Figure\plv_node_num.fig')

% 平均度
figure
hold on;
plot(threshold, sim_degree_origin, 'b-o', 'LineWidth', 1);
plot(threshold, sim_degree_80, 'm--s', 'LineWidth', 1);
plot(threshold, sim_degree_90, 'c-.^', 'LineWidth', 1);
plot(threshold, sim_degree_100, 'k-x', 'LineWidth', 1);
plot(threshold, sim_degree_110, 'r--d', 'LineWidth', 1);
plot(threshold, sim_degree_120, 'g-.p', 'LineWidth', 1);
hold off;
set(gca,'xlim',[0 0.24], 'xtick',0:0.04:0.24,...
        'FontSize',ticklabelsize,'FontName',fontname, 'FontWeight', 'bold')
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Average Degree','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
legend({'Without TMS','80% MT','90% MT','100% MT','110% MT','120% MT'},...
       'FontSize',ticklabelsize,'FontName',fontname, 'Location','best')
legend('boxoff')
saveas(gcf, 'Figure\plv_degree.fig')

% 特征路径长度
figure 
hold on;
plot(threshold, sim_cpl_origin, 'b-o', 'LineWidth', 1);
plot(threshold, sim_cpl_80, 'm--s', 'LineWidth', 1);
plot(threshold, sim_cpl_90, 'c-.^', 'LineWidth', 1);
plot(threshold, sim_cpl_100, 'k-x', 'LineWidth', 1);
plot(threshold, sim_cpl_110, 'r--d', 'LineWidth', 1);
plot(threshold, sim_cpl_120, 'g-.p', 'LineWidth', 1);
hold off;
set(gca,'xlim',[0 0.24], 'xtick',0:0.04:0.24,...
        'FontSize',ticklabelsize,'FontName',fontname, 'FontWeight', 'bold')
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Characteristic Path Length','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
% legend({'Without TMS','80% MT','90% MT','100% MT','110% MT','120% MT'},...
%        'FontSize',ticklabelsize,'FontName',fontname, 'Location','best')
% legend('boxoff')
saveas(gcf, 'Figure\plv_cpl.fig')

% 平均聚类系数
figure 
hold on;
plot(threshold, sim_cc_origin, 'b-o', 'LineWidth', 1);
plot(threshold, sim_cc_80, 'm--s', 'LineWidth', 1);
plot(threshold, sim_cc_90, 'c-.^', 'LineWidth', 1);
plot(threshold, sim_cc_100, 'k-x', 'LineWidth', 1);
plot(threshold, sim_cc_110, 'r--d', 'LineWidth', 1);
plot(threshold, sim_cc_120, 'g-.p', 'LineWidth', 1);
hold off;
set(gca,'xlim',[0 0.24], 'xtick',0:0.04:0.24,...
        'FontSize',ticklabelsize,'FontName',fontname, 'FontWeight', 'bold')
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Clustering Coefficient','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
% legend({'Without TMS','80% MT','90% MT','100% MT','110% MT','120% MT'},...
%        'FontSize',ticklabelsize,'FontName',fontname, 'Location','best')
% legend('boxoff')
saveas(gcf, 'Figure\plv_cc.fig')

% 全局效率
figure 
hold on;
plot(threshold, sim_Eglo_origin, 'b-o', 'LineWidth', 1);
plot(threshold, sim_Eglo_80, 'm--s', 'LineWidth', 1);
plot(threshold, sim_Eglo_90, 'c-.^', 'LineWidth', 1);
plot(threshold, sim_Eglo_100, 'k-x', 'LineWidth', 1);
plot(threshold, sim_Eglo_110, 'r--d', 'LineWidth', 1);
plot(threshold, sim_Eglo_120, 'g-.p', 'LineWidth', 1)
hold off;
set(gca,'xlim',[0 0.24], 'xtick',0:0.04:0.24,...
        'FontSize',ticklabelsize,'FontName',fontname, 'FontWeight', 'bold')
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Global Efficiency','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
% legend({'Without TMS','80% MT','90% MT','100% MT','110% MT','120% MT'},...
%        'FontSize',ticklabelsize,'FontName',fontname, 'Location','best')
% legend('boxoff')
saveas(gcf, 'Figure\plv_Eglo.fig')

figure
hold on;
i=1;
sim_cc_rate_80=sim_cc_80./sim_cc_rand_80;sim_cpl_rate_80=sim_cpl_80./sim_cpl_rand_80;
sim_cc_rate_90=sim_cc_90./sim_cc_rand_90;sim_cpl_rate_90=sim_cpl_90./sim_cpl_rand_90;
sim_cc_rate_100=sim_cc_100./sim_cc_rand_100;sim_cpl_rate_100=sim_cpl_100./sim_cpl_rand_100;
sim_cc_rate_110=sim_cc_110./sim_cc_rand_110;sim_cpl_rate_110=sim_cpl_110./sim_cpl_rand_110;
sim_cc_rate_120=sim_cc_120./sim_cc_rand_120;sim_cpl_rate_120=sim_cpl_120./sim_cpl_rand_120;
% 计算 Small World Index
swi_80(i,:) = sim_cc_rate_80(1,1:24) ./ sim_cpl_rate_80(1,1:24);
swi_90(i,:) = sim_cc_rate_90(1,1:24) ./ sim_cpl_rate_90(1,1:24);
swi_100(i,:) = sim_cc_rate_100(1,1:24) ./ sim_cpl_rate_100(1,1:24);
swi_110(i,:) = sim_cc_rate_110(1,1:24) ./ sim_cpl_rate_110(1,1:24);
swi_120(i,:) = sim_cc_rate_120(1,1:24) ./ sim_cpl_rate_120(1,1:24);
% 找出 threshold 对应的数值，如果为 0，则替换为 NaN
swi_80(swi_80 == 0) = NaN;
swi_90(swi_90 == 0) = NaN;
swi_100(swi_100 == 0) = NaN;
swi_110(swi_110 == 0) = NaN;
swi_120(swi_120 == 0) = NaN;
% 绘图
plot(threshold(1:24), swi_80(i,:), 'm--s','LineWidth', 1);
plot(threshold(1:24), swi_90(i,:), 'c-.^','LineWidth', 1);
plot(threshold(1:24), swi_100(i,:), 'k-x','LineWidth', 1);
plot(threshold(1:24), swi_110(i,:), 'r--d','LineWidth', 1);
plot(threshold(1:24), swi_120(i,:), 'g-.p','LineWidth', 1);
plot(threshold(1:22), sim_cc_origin(1:22) ./ sim_cc_origin_rand(1:22) ./ (sim_cpl_origin(1:22) ./ sim_cpl_origin_rand(1:22)), 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'ylim', [1 1.2], 'ytick', 1:0.05:1.2, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Small World Index', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
saveas(gcf, 'Figure/small_world_index.fig');

% 其他图形（以聚类系数为例，其余类似修改）
figure 
plot(threshold, sim_cc_origin, 'b-o', threshold, sim_cc_origin_rand, 'b--o')
hold on
plot(threshold, sim_cc_80, 'm--s', threshold, sim_cc_rand_80, 'm:^')
plot(threshold, sim_cc_90, 'c-.^', threshold, sim_cc_rand_90, 'c-.+')
plot(threshold, sim_cc_100, 'k-x', threshold, sim_cc_rand_100, 'k-.+')
plot(threshold, sim_cc_110, 'r--d', threshold, sim_cc_rand_110, 'r--v')
plot(threshold, sim_cc_120, 'g-.p', threshold, sim_cc_rand_120, 'g--h')
hold off
legend({'Without TMS','Without TMS Rand','80% MT','80% Rand','90% MT','90% Rand','100% MT','100% Rand','110% MT','110% Rand','120% MT','120% Rand'},...
       'FontSize',ticklabelsize-2,'FontName',fontname, 'NumColumns',2)
set(gca,'xlim',[0 0.24], 'xtick',0:0.04:0.24,...
        'FontSize',ticklabelsize,'FontName',fontname)
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Clustering Coefficient','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
legend('boxoff')
saveas(gcf, 'Figure\plv_cc_rand.fig')

figure 
load('Data\graph_theoretical_plv.mat')
plot(threshold, sim_cpl_origin, 'b-o', threshold, sim_cpl_origin_rand, 'b--o')
hold on
plot(threshold, sim_cpl_80, 'm--s', threshold, sim_cpl_rand_80, 'm:^')
plot(threshold, sim_cpl_90, 'c-.^', threshold, sim_cpl_rand_90, 'c-.+')
plot(threshold, sim_cpl_100, 'k-x', threshold, sim_cpl_rand_100, 'k-.+')
plot(threshold, sim_cpl_110, 'r--d', threshold, sim_cpl_rand_110, 'r--v')
plot(threshold, sim_cpl_120, 'g-.p', threshold, sim_cpl_rand_120, 'g--h')
hold off
legend({'Without TMS','Without TMS Rand','80% MT','80% Rand','90% MT','90% Rand','100% MT','100% Rand','110% MT','110% Rand','120% MT','120% Rand'},...
       'FontSize',ticklabelsize-2,'FontName',fontname, 'NumColumns',2)
set(gca,'xlim',[0 0.24], 'xtick',0:0.04:0.24,...
        'FontSize',ticklabelsize,'FontName',fontname)
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Characteristic Path Length','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
legend('boxoff')
saveas(gcf, 'Figure\plv_cpl_rand.fig')

figure 
load('Data\graph_theoretical_plv.mat')
plot(threshold, sim_cc_origin./sim_cc_origin_rand, 'b-o', 'LineWidth', 1,...
     threshold, sim_cc_80./sim_cc_rand_80, 'm--s', 'LineWidth', 1,...
     threshold, sim_cc_90./sim_cc_rand_90, 'c-.^', 'LineWidth', 1,...
     threshold, sim_cc_100./sim_cc_rand_100, 'k-x', 'LineWidth', 1,...
     threshold, sim_cc_110./sim_cc_rand_110, 'r--d', 'LineWidth', 1,...
     threshold, sim_cc_120./sim_cc_rand_120, 'g-.p', 'LineWidth', 1)
legend({'Without TMS','80% MT','90% MT','100% MT','110% MT','120% MT'},...
       'FontSize',ticklabelsize,'FontName',fontname, 'Location','best')
set(gca,'xlim',[0 0.24], 'xtick',0:0.04:0.24,...
        'FontSize',ticklabelsize,'FontName',fontname)
xlabel('Threshold','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
ylabel('Clustering Coefficient Rate','FontSize',fontsize,'FontName',fontname, 'FontWeight', 'bold')
legend('boxoff')
saveas(gcf, 'Figure\plv_cc_rate.fig')
%% 图论(all)
load('Data\Cij.mat')
load('Data\sim_plv_all.mat')
load('Data\sim_plv_origin.mat')
sim_degree=[];sim_Eglo=[];sim_cpl=[];sim_cc=[];
sim_degree_origin=[];sim_Eglo_origin=[];sim_cpl_origin=[];sim_cc_origin=[];
sim_degree_rand=[];sim_Eglo_rand=[];sim_cpl_rand=[];sim_cc_rand=[];
sim_degree_rand_origin=[];sim_Eglo_rand_origin=[];sim_cpl_rand_origin=[];sim_cc_rand_origin=[];
sim_plv_nodenum=[];sim_plv_nodenum_origin=[];
strength = strengths_und(Cij);
threshold = 0:0.01:0.5;
for j = 1:41
    n = 1;
    for i = threshold
        sim_plv_thre(:,:,j) = threshold_absolute(sim_plv_all(:,:,j),i);
        sim_plv_origin_thre = threshold_absolute(sim_plv_origin,i);
        [sim_degree(j,n),sim_Eglo(j,n),sim_cpl(j,n),sim_cc(j,n)] = anal_cm(sim_plv_thre(:,:,j));
        [sim_degree_origin(n),sim_Eglo_origin(n),sim_cpl_origin(n),sim_cc_origin(n)] = anal_cm(sim_plv_origin_thre);
        sim_plv_thre_rand(:,:,j)=makerandCIJ_und(size(sim_plv_thre(:,:,j),1),size(sim_plv_thre(sim_plv_thre(:,:,j)~=0),1)/2);
        sim_plv_origin_thre_rand=makerandCIJ_und(size(sim_plv_origin_thre,1),size(sim_plv_origin_thre(sim_plv_origin_thre~=0),1)/2);
        sim_plv_thre_rand(:,:,j)=randmio_und(sim_plv_thre_rand(:,:,j),10);
        sim_plv_origin_thre_rand=randmio_und(sim_plv_origin_thre_rand,10);
        [~,~,sim_cpl_rand(j,n),sim_cc_rand(j,n)] = anal_cm(sim_plv_thre_rand(:,:,j));
        [~,~,sim_cpl_origin_rand(n),sim_cc_origin_rand(n)] = anal_cm(sim_plv_origin_thre_rand);
        sim_plv_nodenum(j,n) = length(find(sum(sim_plv_thre(:,:,j))));
        sim_plv_origin_nodenum(n) = length(find(sum(sim_plv_origin_thre)));
        n = n+1;
    end
    sim_threshold(:,j) = Get_threhold(sim_plv_all(:,:,j));
    sim_origin_threshold = Get_threhold(sim_plv_origin);
    % a = sim_cc(j,n)./sim_cc_rand(j,n);b=sim_cpl(j,n)./sim_cpl_rand(j,n);
    % e = sim_cc_origin./sim_cc_origin_rand;f=sim_cpl_origin./sim_cpl_origin_rand;
end
save('Data\graph_theoretical_plv.mat','threshold','strength','sim_degree','sim_Eglo','sim_cpl',...
    'sim_cc','sim_degree_origin','sim_Eglo_origin', ...
    'sim_cpl_origin','sim_cc_origin', 'sim_threshold','sim_origin_threshold','D','C_origin',...
    'sim_plv_nodenum','sim_plv_origin_nodenum','sim_cpl_rand','sim_cc_rand', ...
    'sim_cpl_origin_rand','sim_cc_origin_rand')
%% 图论绘制(all)已弃用
% 载入数据
load('Data/graph_theoretical_plv.mat')
fontsize=12;
ticklabelsize = 10;
fontname='Times New Roman';
% 删除emp相关数据，仅保留origin和sim数据
thresholds = threshold; % 提取阈值数据
num_strengths = size(sim_plv_nodenum, 1); % 41个强度点
num_thresholds = size(sim_plv_nodenum, 2); % 51个阈值点

% 设定颜色映射，每10个强度点使用一种颜色
colors = lines(5); % 5种颜色
color_groups = ceil((1:num_strengths)/10); % 生成颜色组索引

% 创建legend标签
legend_labels = {'80-90%MT', '90-100%MT', '100-110%MT', '110-120%MT', 'Without TMS'};
legend_handles = gobjects(1, 5);

% 节点数作图
figure;
hold on;
for i = 1:num_strengths
    if i == 41
        plot(thresholds, sim_plv_nodenum(i,:), 'Color', colors(color_groups(i-1), :),'LineWidth', 1); 
        break;
    end
    h = plot(thresholds, sim_plv_nodenum(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1 && i ~= 41 % 仅存储第一个出现的颜色作为legend
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(thresholds, sim_plv_origin_nodenum, 'g-o', 'LineWidth', 1); % origin 用蓝色
hold off;
set(gca, 'xlim', [0 0.5], 'xtick', 0:0.1:0.5, 'ylim', [50 70], 'ytick', 50:2:70, 'FontSize', ticklabelsize, 'FontName', fontname,'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname,'FontWeight', 'bold');
ylabel('Node number', 'FontSize', fontsize, 'FontName', fontname,'FontWeight', 'bold', 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname,'FontWeight', 'bold', 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/plv_node_num.fig');

% 平均度作图
figure;
hold on;
for i = 1:num_strengths
    h = plot(thresholds, sim_degree(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(thresholds, sim_degree_origin, 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Average Degree', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/plv_degree.fig');

% 特征路径长度作图
figure;
hold on;
for i = 1:num_strengths
    h = plot(thresholds, sim_cpl(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(thresholds, sim_cpl_origin, 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Characteristic Path Length', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/plv_cpl.fig');

% 平均聚类系数作图
figure;
hold on;
for i = 1:num_strengths
    h = plot(thresholds, sim_cc(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(thresholds, sim_cc_origin, 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Clustering Coefficient', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/plv_cc.fig');

% 全局效率作图
figure;
hold on;
for i = 1:num_strengths
    h = plot(thresholds, sim_Eglo(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(thresholds, sim_Eglo_origin, 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Global Efficiency', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/plv_Eglo.fig');

% 平均聚类系数
figure; hold on;
for i = 1:num_strengths
    h = plot(threshold, sim_cc(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(threshold, sim_cc_origin, 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Clustering Coefficient', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/plv_cc_rand.fig');

% 特征路径长度
figure; hold on;
for i = 1:num_strengths
    h = plot(threshold, sim_cpl(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(threshold, sim_cpl_origin, 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Characteristic Path Length', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/plv_cpl_rand.fig');

% 小世界指数
figure; hold on;
sim_cc_rate=sim_cc./sim_cc_rand;sim_cpl_rate=sim_cpl./sim_cpl_rand;
for i = 1:num_strengths
    % 计算 Small World Index
    swi(i,:) = sim_cc_rate(i,1:24) ./ sim_cpl_rate(i,1:24);
    % 找出 threshold 对应的数值，如果为 0，则替换为 NaN
    swi(swi == 0) = NaN;
    % 绘图
    h = plot(threshold(1:24), swi(i,:), 'Color', colors(color_groups(i), :),'LineWidth', 1);
    if mod(i, 10) == 1
        legend_handles(color_groups(i)) = h;
    end
end
legend_handles(end) = plot(threshold(1:22), sim_cc_origin(1:22) ./ sim_cc_origin_rand(1:22) ./ (sim_cpl_origin(1:22) ./ sim_cpl_origin_rand(1:22)), 'g-o', 'LineWidth', 1);
hold off;
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'ylim', [1 1.2], 'ytick', 1:0.05:1.2, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Small World Index', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
legend(legend_handles, legend_labels, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
legend('boxoff');
saveas(gcf, 'Figure/small_world_index.fig');
%% 图论绘制(all)
% 载入数据
load('Data/graph_theoretical_plv.mat')
fontsize = 12;
ticklabelsize = 10;
fontname = 'Times New Roman';

% 设置颜色映射
num_strengths = 41; % 80-120%MT共41个点
stim_strengths = linspace(80, 120, num_strengths); % 刺激强度值
color_map = jet(num_strengths); % 使用jet颜色映射
withoutTMS_color = [0.39,0.83,0.07]; % 绿色(RGB)

% 节点数作图
figure;
hold on;
% 绘制所有刺激强度曲线
for i = 1:num_strengths
    plot(threshold, sim_plv_nodenum(i,:), 'Color', color_map(i,:), 'LineWidth', 1);
end
% 绘制withoutTMS曲线
h_without = plot(threshold, sim_plv_origin_nodenum, 'Color', withoutTMS_color,...
                'LineWidth', 1, 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 6);
% 坐标轴设置
set(gca, 'xlim', [0 0.5], 'ylim', [50 70], 'FontSize', ticklabelsize,...
         'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Node number', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
% 添加颜色条
colormap(jet);
clim([80 120]);
cb = colorbar('Ticks', 80:10:120, 'TickLabels', {'80%', '90%', '100%', '110%', '120%'},...
             'FontSize', ticklabelsize, 'FontName', fontname);
cb.Label.String = 'Stimulation Intensity (MT)';
cb.Label.FontSize = fontsize;
% 添加图例
% legend(h_without, 'Without TMS', 'FontSize', ticklabelsize, 'FontName', fontname,...
%        'Location', 'best', 'Box', 'off');
saveas(gcf, 'Figure/plv_node_num.fig');
hold off;

% 平均度作图
figure;
hold on;
for i = 1:num_strengths
    plot(threshold, sim_degree(i,:), 'Color', color_map(i,:), 'LineWidth', 1);
end
h_without = plot(threshold, sim_degree_origin, 'Color', withoutTMS_color,...
                'LineWidth', 1, 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 6);
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Average Degree', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
colormap(jet);
clim([80 120]);
cb = colorbar('Ticks', 80:10:120, 'TickLabels', {'80%', '90%', '100%', '110%', '120%'},...
             'FontSize', ticklabelsize, 'FontName', fontname);
cb.Label.String = 'Stimulation Intensity (MT)';
cb.Label.FontSize = fontsize;
% legend(h_without, 'Without TMS', 'FontSize', ticklabelsize, 'FontName', fontname,...
%        'Location', 'best', 'Box', 'off');
saveas(gcf, 'Figure/plv_degree.fig');
hold off;

% 特征路径长度作图
figure;
hold on;
for i = 1:num_strengths
    plot(threshold, sim_cpl(i,:), 'Color', color_map(i,:), 'LineWidth', 1);
end
h_without = plot(threshold, sim_cpl_origin, 'Color', withoutTMS_color,...
                'LineWidth', 1, 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 6);
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Characteristic Path Length', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
colormap(jet);
clim([80 120]);
cb = colorbar('Ticks', 80:10:120, 'TickLabels', {'80%', '90%', '100%', '110%', '120%'},...
             'FontSize', ticklabelsize, 'FontName', fontname);
cb.Label.String = 'Stimulation Intensity (MT)';
cb.Label.FontSize = fontsize;
% legend(h_without, 'Without TMS', 'FontSize', ticklabelsize, 'FontName', fontname,...
%        'Location', 'best', 'Box', 'off');
saveas(gcf, 'Figure/plv_cpl.fig');
hold off;

% 平均聚类系数作图
figure;
hold on;
for i = 1:num_strengths
    plot(threshold, sim_cc(i,:), 'Color', color_map(i,:), 'LineWidth', 1);
end
h_without = plot(threshold, sim_cc_origin, 'Color', withoutTMS_color,...
                'LineWidth', 1, 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 6);
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Clustering Coefficient', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
colormap(jet);
clim([80 120]);
cb = colorbar('Ticks', 80:10:120, 'TickLabels', {'80%', '90%', '100%', '110%', '120%'},...
             'FontSize', ticklabelsize, 'FontName', fontname);
cb.Label.String = 'Stimulation Intensity (MT)';
cb.Label.FontSize = fontsize;
% legend(h_without, 'Without TMS', 'FontSize', ticklabelsize, 'FontName', fontname,...
%        'Location', 'best', 'Box', 'off');
saveas(gcf, 'Figure/plv_cc.fig');
hold off;

% 全局效率作图
figure;
hold on;
for i = 1:num_strengths
    plot(threshold, sim_Eglo(i,:), 'Color', color_map(i,:), 'LineWidth', 1);
end
h_without = plot(threshold, sim_Eglo_origin, 'Color', withoutTMS_color,...
                'LineWidth', 1, 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 6);
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24,'FontSize', ticklabelsize, 'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Global Efficiency', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
colormap(jet);
clim([80 120]);
cb = colorbar('Ticks', 80:10:120, 'TickLabels', {'80%', '90%', '100%', '110%', '120%'},...
             'FontSize', ticklabelsize, 'FontName', fontname);
cb.Label.String = 'Stimulation Intensity (MT)';
cb.Label.FontSize = fontsize;
% legend(h_without, 'Without TMS', 'FontSize', ticklabelsize, 'FontName', fontname,...
%        'Location', 'best', 'Box', 'off');
saveas(gcf, 'Figure/plv_Eglo.fig');
hold off;

% 小世界指数作图
figure;
hold on;
sim_cc_rate=sim_cc./sim_cc_rand;sim_cpl_rate=sim_cpl./sim_cpl_rand;
% 计算并绘制所有刺激强度曲线
for i = 1:num_strengths
    swi = sim_cc_rate(i,1:24) ./ sim_cpl_rate(i,1:24);
    swi(swi == 0) = NaN;
    plot(threshold(1:24), swi, 'Color', color_map(i,:), 'LineWidth', 1);
end
% 绘制withoutTMS曲线
swi_origin = sim_cc_origin(1:22) ./ sim_cc_origin_rand(1:22) ./...
            (sim_cpl_origin(1:22) ./ sim_cpl_origin_rand(1:22));
h_without = plot(threshold(1:22), swi_origin, 'Color', withoutTMS_color,...
                'LineWidth', 1, 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 6);
% 坐标轴设置
set(gca, 'xlim', [0 0.24], 'xtick', 0:0.04:0.24, 'ylim', [1 1.2], 'FontSize', ticklabelsize,...
         'FontName', fontname, 'FontWeight', 'bold');
xlabel('Threshold', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
ylabel('Small World Index', 'FontSize', fontsize, 'FontName', fontname, 'FontWeight', 'bold');
% 颜色条
colormap(jet);
clim([80 120]);
cb = colorbar('Ticks', 80:10:120, 'TickLabels', {'80%', '90%', '100%', '110%', '120%'},...
             'FontSize', ticklabelsize, 'FontName', fontname);
cb.Label.String = 'Stimulation Intensity (MT)';
cb.Label.FontSize = fontsize;
% 图例
% legend(h_without, 'Without TMS', 'FontSize', ticklabelsize, 'FontName', fontname,...
%        'Location', 'best', 'Box', 'off');
saveas(gcf, 'Figure/small_world_index.fig');
hold off;