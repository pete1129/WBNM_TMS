% -----------------------
% 计算模拟数据相关分析指标
% -----------------------
%case 1 不同全局耦合系数的皮尔森相关性和平滑滤波曲线(矩阵间相关性定义为下三角/上三角元素间的相关性)
%case 2 不同全局耦合系数的锁相值相关性和平滑滤波曲线
%case 3 不同全局耦合系数的各脑区基线电位
%case 4 不同全局耦合系数的各脑区主频率
%case 5 模拟点处的各脑区基线电位
%case 6 模拟点处的各脑区时间序列、功率谱密度、主频率
%case 7 模拟点处的皮尔森相关系数功能连接矩阵
%case 8 模拟点处的锁相值功能连接矩阵
%case 9 计算不同数量trial时皮尔森相关性、锁相值相关性
%case 10 计算alpha脑区的个数
%case 11 计算corr图论分析指标
%case 12 计算plv图论分析指标
%case 13 不同阈值下，相关性和相似性的变化
%2-8-15-16-12-13
clc;clear;close all;
metric = 14;
fs = 1000;
nTrial = 50;
expnum = dir(['Expfile\' '*.txt']);

switch metric
    
    case 1 % 不同全局耦合系数的皮尔森相关性和平滑滤波曲线
        load('Data\emp_fc.mat')
        a = [];correlation_coefficient = [];D = 0.8:0.01:1.2;
        % for n = 81 : (80 + length(expnum))
        for n = 204   
            nTrial = 50;
            expName = ['exp' num2str(n, '%03d')];
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                a(:,:,i) = corr(detrend(excs(1001:2001,:)));
            end
            correlation_coefficient(n-80) = cal_fc_correlation(mean(a,3),emp_fc);
        end
        [M,I]=max(correlation_coefficient);
        corr_fitdata = smoothdata(correlation_coefficient,'sgolay',41);
        save('Data\correlation_coefficient.mat','correlation_coefficient','corr_fitdata','D','M','I')
    
    case 2 % 不同全局耦合系数的锁相值相关性和平滑滤波曲线
        load('Data\emp_plv.mat')
        a = [];phase_locking_value = [];excs_all = [];D = 0.8:0.01:1.2;
        % for n = 81 : (80 + length(expnum))
        for n = 204 
            expName = ['exp' num2str(n, '%03d')];
            nTrial = 50;
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                excs_all(:,:,i) = excs;
                a(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));          
            end
            phase_locking_value(n-80) = cal_fc_correlation(mean(a,3),emp_plv);
            excs = mean(excs_all,3);
        end
        [M,I]=max(phase_locking_value);
        plv_fitdata = smoothdata(phase_locking_value,'sgolay',41);
        save('Data\phase_locking_value.mat','phase_locking_value','plv_fitdata','D','M','I')
        save('Data/excs.mat','excs')
        % save('Data/excs_all.mat','excs_all')
    case 3 % 不同全局耦合系数的各脑区基线电位
        b = [];baseline_potential = [];D = 80:1:120;
        for n = 81 : (80 + length(expnum))
            expName = ['exp' num2str(n, '%03d')];
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                b(i,:) = mean(excs(1001:1200,:));
            end
            baseline_potential(n-80,:) = mean(b);
        end
        save('Data\baseline_potential_200.mat','baseline_potential','D')
    
    case 4 % 不同全局耦合系数的各脑区主频率
        c = [];dominant_rhythm = [];D=80:1:120;
        for n = 81 : (80 + length(expnum))
        % n = 203;
            expName = ['exp' num2str(n, '%03d')];
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                x = detrend(excs(1801:2000,:));
                N = length(x);
                nfft = 2^nextpow2(N);
                [per,f] = periodogram(x,[],nfft,fs);
                [M,I] = max(per);
                for j = 1 : 68
                    c(i,j) =f(I(j));
                end
            end
            dominant_rhythm(n-80,:) = mean(c);    
        end
        save('Data\dominant_rhythm_800.mat','dominant_rhythm','D')
    
    case 5 % 模拟点处的各脑区基线电位
        load('Data\correlation_coefficient.mat')
        load('Data\phase_locking_value.mat')
        b = [];I=[];sim_baseline_potential=[];
        [M,I(1)] = max(correlation_coefficient);
        [M,I(2)] = max(phase_locking_value);
        DD=[];DD(1)=D(I(1));DD(2)=D(I(2));D=[];D=DD;
        for n = 81:(80 + length(D))
            expName = ['exp' num2str(I(n), '%03d')];
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                b(i,:) = mean(excs(1001:2001,:));
            end
            sim_baseline_potential(n,:) = mean(b);
         end
        save('Data\sim_baseline_potential.mat','sim_baseline_potential','D')
    
    case 6 % 模拟点处的各脑区时间序列、功率谱密度、主频率
        load('Data\correlation_coefficient.mat')
        load('Data\phase_locking_value.mat')
        c = [];In=[];sim_dominant_rhythm=[];sim_excs=[];sim_psd=[];sim_ns=[];D=80:1:120;
        % [M,In(1)] = max(correlation_coefficient);
        % [M,In(2)] = max(phase_locking_value);
        % DD=[];DD(1)=D(In(1));DD(2)=D(In(2));D=[];D=DD;
        % for n = 81 : (80+length(D))
            % expName = ['exp' num2str(In(n), '%03d')];
        for n = 204
            expName = ['exp' num2str(n, '%03d')];
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                x = detrend(excs(1001:2001,:));
                N = length(x);
                nfft = 2^nextpow2(N);
                [per,f] = periodogram(x,[],nfft,fs);
                [M,I] = max(per);
                for j = 1 : 68
                    c(i,j) =f(I(j));
                end
            end
            for k=1:68
                [~,nspectrum(:,k)] = nom_spectrum(excs(1001:2001,k),fs);
            end
            t=0:0.001:1.5;
            sim_dominant_rhythm(1,:) = mean(c);
            % sim_excs(:,:,n-80) = excs;
            % sim_psd(:,:,n-80) = per;
            % sim_ns(:,:,n-80) = nspectrum;
            sim_excs(:,:,n) = excs;
            sim_psd(:,:,n) = per;
            sim_ns(:,:,n) = nspectrum;
        end
        save('Data\sim_excs.mat','sim_excs','D','t')
        save('Data\sim_psd.mat','sim_psd','D','f')
        save('Data\sim_ns.mat','sim_ns','D','f')
        save('Data\sim_dominant_rhythm.mat','sim_dominant_rhythm','D')
    
    case 7 % 模拟点处的皮尔森相关系数功能连接矩阵
        load('Data\correlation_coefficient.mat')
        nTrial = 50;
        sim_fc = [];
        DD=D(I);D=[];D=DD;
        expName = ['exp' num2str(I, '%03d')];
        for i = 1 : nTrial
            loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
            load(loadfilename)
            sim_fc(:,:,i) = corr(detrend(excs(2001:3001,:)));
        end
        sim_fc = mean(sim_fc,3);
        average_fc = mean(sim_fc(:));
        save('Data\sim_fc.mat','sim_fc','D','average_fc')
    
    case 8 % 模拟点处的锁相值功能连接矩阵
        % load('Data\phase_locking_value.mat')
        % sim_plv = [];
        % DD=D(I);D=[];D=DD;
        % expName = ['exp' num2str(I+79, '%03d')];
        % nTrial = 50;
        % for i = 1: nTrial
        %     loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
        %     load(loadfilename)
        %     sim_plv(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));
        % end
        % sim_plv = mean(sim_plv,3);
        % average_value = mean(sim_plv(:));
        % save('Data\sim_plv.mat','sim_plv','D','average_value')
        n=204;
        load('Data\phase_locking_value.mat')
        sim_plv = [];
        DD=D(I);D=[];D=DD;
        expName = ['exp' num2str(n, '%03d')];
        nTrial = 20;
        for i = 1: nTrial
            loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
            load(loadfilename)
            sim_plv(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));
        end 
        sim_plv = mean(sim_plv,3);
        average_value = mean(sim_plv(:));
        save('Data\sim_plv.mat','sim_plv','D')
        % for i = 1:68
        %     sim_plv(i, i) = 0;
        % end
        % average_value = mean(sim_plv(:));
        % save('Data\sim_plv.mat','sim_plv','D')
    
    case 9 % 计算不同数量trial时皮尔森相关性、锁相值相关性
        load('Data\emp_fc.mat')
        load('Data\emp_plv.mat')
        load('Data\correlation_coefficient.mat')
        load('Data\phase_locking_value.mat')
        e = [];In=[];correlation_coefficient_ntrial = [];phase_locking_value_ntrial = [];ntrial=10:10:50;
        [M,In(1)] = max(correlation_coefficient);
        [M,In(2)] = max(phase_locking_value);
        DD=[];DD(1)=D(In(1));DD(2)=D(In(2));
        expName = ['exp' num2str(In(1), '%03d')];
        for i = 1 : 50
            loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
            load(loadfilename)
            e(:,:,i) = corr(detrend(excs(1001:2000,:)));
        end
        for n = 1 : 3
            correlation_coefficient_ntrial(n) = cal_fc_correlation(mean(e(:,:,1:10*n),3),emp_fc);
        end
        D=[];D=DD(1);
        save('Data\correlation_coefficient_ntrial.mat','correlation_coefficient_ntrial','D','ntrial')
        
        expName = ['exp' num2str(In(2), '%03d')];e=[];
        for i = 1 : 50
            loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
            load(loadfilename)
            e(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2000,:)));
        end
        for n = 1 : 3
            phase_locking_value_ntrial(n) = cal_fc_correlation(mean(e(:,:,1:10*n),3),emp_plv);
        end
        D=[];D=DD(2);
        save('Data\phase_locking_value_ntrial.mat','phase_locking_value_ntrial','D','ntrial')
    
    case 10 % 计算alpha脑区的个数
        load('Data\dominant_rhythm.mat')
        load('Data\correlation_coefficient.mat')
        load('Data\phase_locking_value.mat')
        I=[];
        [M,I(1)] = max(correlation_coefficient);
        [M,I(2)] = max(phase_locking_value);
        DD=[];DD(1)=D(I(1));DD(2)=D(I(2));D=[];D=DD;DD=[];DD=0.8:0.01:1.2;
        dominant_rhythm(dominant_rhythm>8&dominant_rhythm<12) = 1;
        dominant_rhythm(dominant_rhythm~=1) = 0;
        alpha_rhythm_nums = sum(dominant_rhythm,2);
        alpha_rhythm_num(1) = alpha_rhythm_nums(I(1));
        alpha_rhythm_num(2) = alpha_rhythm_nums(I(2));
        save('Data\alpha_rhythm_nums.mat','D','DD','alpha_rhythm_num','alpha_rhythm_nums')
    
    case 11 % 计算corr图论分析指标
        load('Data\Cij.mat')
        load('Data\sim_fc.mat')
        load('Data\emp_fc.mat')
        sim_degree=[];sim_Eglo=[];sim_cpl=[];sim_cc=[];
        emp_degree=[];emp_Eglo=[];emp_cpl=[];emp_cc=[];
        sim_fc_nodenum=[];emp_fc_nodenum=[];
        strength = strengths_und(Cij);
        threshold = 0:0.01:1;n = 1;
        sim_fc = abs(sim_fc);
        for i = threshold
            sim_fc_thre = threshold_absolute(sim_fc,i);
            emp_fc_thre = threshold_absolute(emp_fc,i);
            [sim_degree(n),sim_Eglo(n),sim_cpl(n),sim_cc(n)] = anal_cm(sim_fc_thre);
            [emp_degree(n),emp_Eglo(n),emp_cpl(n),emp_cc(n)] = anal_cm(emp_fc_thre);
            sim_fc_nodenum(n) = length(find(sum(sim_fc_thre)));
            emp_fc_nodenum(n) = length(find(sum(emp_fc_thre)));
            n = n+1;
        end
        sim_threshold = Get_threhold(sim_fc);
        emp_threshold = Get_threhold(emp_fc);
        save('Data\graph_theoretical_fc.mat','threshold','strength','sim_degree','sim_Eglo','sim_cpl',...
            'sim_cc','emp_degree','emp_Eglo','emp_cpl','emp_cc','sim_threshold','emp_threshold','D','sim_fc_nodenum','emp_fc_nodenum')
    
    case 12 % 计算plv图论分析指标
        load('Data\Cij.mat')
        load('Data\sim_plv.mat')
        load('Data\emp_plv.mat')
        load('Data\sim_plv_origin.mat')
        sim_degree=[];sim_Eglo=[];sim_cpl=[];sim_cc=[];
        emp_degree=[];emp_Eglo=[];emp_cpl=[];emp_cc=[];
        sim_degree_origin=[];sim_Eglo_origin=[];sim_cpl_origin=[];sim_cc_origin=[];
        sim_degree_rand=[];sim_Eglo_rand=[];sim_cpl_rand=[];sim_cc_rand=[];
        emp_degree_rand=[];emp_Eglo_rand=[];emp_cpl_rand=[];emp_cc_rand=[];
        sim_degree_rand_origin=[];sim_Eglo_rand_origin=[];sim_cpl_rand_origin=[];sim_cc_rand_origin=[];
        sim_plv_nodenum=[];emp_plv_nodenum=[];sim_plv_nodenum_origin=[];
        strength = strengths_und(Cij);
        threshold = 0:0.01:0.5;
        n = 1;
        for i = threshold
            sim_plv_thre = threshold_absolute(sim_plv,i);
            emp_plv_thre = threshold_absolute(emp_plv,i);
            sim_plv_origin_thre = threshold_absolute(sim_plv_origin,i);
            [sim_degree(n),sim_Eglo(n),sim_cpl(n),sim_cc(n)] = anal_cm(sim_plv_thre);
            [emp_degree(n),emp_Eglo(n),emp_cpl(n),emp_cc(n)] = anal_cm(emp_plv_thre);
            [sim_degree_origin(n),sim_Eglo_origin(n),sim_cpl_origin(n),sim_cc_origin(n)] = anal_cm(sim_plv_origin_thre);
            sim_plv_thre_rand=makerandCIJ_und(size(sim_plv_thre,1),size(sim_plv_thre(sim_plv_thre~=0),1)/2);
            emp_plv_thre_rand=makerandCIJ_und(size(emp_plv_thre,1),size(emp_plv_thre(emp_plv_thre~=0),1)/2);
            sim_plv_origin_thre_rand=makerandCIJ_und(size(sim_plv_origin_thre,1),size(sim_plv_origin_thre(sim_plv_origin_thre~=0),1)/2);
            sim_plv_thre_rand=randmio_und(sim_plv_thre_rand,10);
            emp_plv_thre_rand=randmio_und(emp_plv_thre_rand,10);
            sim_plv_origin_thre_rand=randmio_und(sim_plv_origin_thre_rand,10);
            [~,~,sim_cpl_rand(n),sim_cc_rand(n)] = anal_cm(sim_plv_thre_rand);
            [~,~,emp_cpl_rand(n),emp_cc_rand(n)] = anal_cm(emp_plv_thre_rand);
            [~,~,sim_cpl_origin_rand(n),sim_cc_origin_rand(n)] = anal_cm(sim_plv_origin_thre_rand);
            sim_plv_nodenum(n) = length(find(sum(sim_plv_thre)));
            emp_plv_nodenum(n) = length(find(sum(emp_plv_thre)));
            sim_plv_origin_nodenum(n) = length(find(sum(sim_plv_origin_thre)));
            n = n+1;
        end
        sim_threshold = Get_threhold(sim_plv);
        emp_threshold = Get_threhold(emp_plv);
        sim_origin_threshold = Get_threhold(sim_plv_origin);
        a = sim_cc./sim_cc_rand;b=sim_cpl./sim_cpl_rand;
        c = emp_cc./emp_cc_rand;d=emp_cpl./emp_cpl_rand;
        e = sim_cc_origin./sim_cc_origin_rand;f=sim_cpl_origin./sim_cpl_origin_rand;
        save('Data\graph_theoretical_plv.mat','threshold','strength','sim_degree','sim_Eglo','sim_cpl',...
            'sim_cc','emp_degree','emp_Eglo','emp_cpl','emp_cc','sim_degree_origin','sim_Eglo_origin', ...
            'sim_cpl_origin','sim_cc_origin', 'sim_threshold','emp_threshold','sim_origin_threshold','D','C_origin',...
            'sim_plv_nodenum','emp_plv_nodenum','sim_plv_origin_nodenum','sim_cpl_rand','sim_cc_rand','emp_cpl_rand', ...
            'emp_cc_rand','sim_cpl_origin_rand','sim_cc_origin_rand')
    
    case 13 % 不同阈值下相关性和相似性的变化
%         load('Data\siminet_data.mat')
        load('Data\sim_plv.mat')
        load('Data\emp_plv.mat')
        load('Data\sim_plv_origin.mat')
        threshold = 0:0.01:0.5;n=1;
        for i = threshold
            sim_plv_thre = threshold_absolute(sim_plv,i);
            emp_plv_thre = threshold_absolute(emp_plv,i);
            sim_plv_origin_thre = threshold_absolute(sim_plv_origin,i);
            matrice_sim_plv = weight_conversion(sim_plv_thre, 'binarize');
            matrice_emp_plv = weight_conversion(emp_plv_thre, 'binarize');
            matrice_sim_plv_origin = weight_conversion(sim_plv_origin_thre, 'binarize');
            phase_locking_value_thre(n) = cal_fc_correlation(sim_plv_thre,emp_plv_thre);
            phase_locking_value_origin_thre(n) = cal_fc_correlation(sim_plv_origin_thre,emp_plv_thre);
%             [similarity_plv(n),distance_plv(n)]=siminet(matrice_sim_plv,matrice_emp_plv,tract,radius);
            n=n+1;
        end
        save('Data\phase_locking_value_thre.mat','phase_locking_value_thre','phase_locking_value_origin_thre','threshold')
    case 14 % 模拟点处的动态功能连接矩阵
        sim_plv = [];
        sim_plv_left = [];
        sim_plv_right = [];
        sim_plv_roi = [];
        D = 80:1:120;
        % D = 204;
        nTrial = 20;

        % 定义左右半球索引 (基于 Desikan-Killiany 图谱)
        left_indices = 1:34;
        right_indices = 35:68;

        % 定义特定 ROI 的索引（例如 DLPFC 左右半球）
        roi_indices = [9,11,13,15,22,24,25]; % DMN
        % roi_indices = [1,9,11,13,14,18,22,24,31]; % DMN(NA)
        % roi_indices = [2,8,14,17,18,19]; % DAN
        % roi_indices = [4,6,10,12]; % VIS
        % roi_indices = [3,26,30]; % SAN
        % roi_indices = 29; % AUD
        % roi_indices = [2,3,17,18,19,25,26,31]; % ECN
        for n = 81 : 5: (80 + length(D))
        % for n = 204 
            expName = ['exp' num2str(n, '%03d')];

            for i = 1:nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat', expName, i);
                load(loadfilename) % 加载 excs 数据

                % 计算整体 PLV
                sim_plv(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));

                % 计算左脑 PLV
                sim_plv_left(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001, left_indices)));

                % 计算右脑 PLV
                sim_plv_right(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001, right_indices)));

                % 计算特定 ROI PLV
                sim_plv_roi(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001, roi_indices)));
            end
            % for i = 1:nTrial
            %     sim_plv(:,:,i) = sim_plv(:,:,i) - diag(diag(sim_plv(:,:,i)));
                % sim_plv_left(:,:,i) = sim_plv_left(:,:,i) - diag(diag(sim_plv_left(:,:,i)));
            %     sim_plv_right(:,:,i) = sim_plv_right(:,:,i) - diag(diag(sim_plv_right(:,:,i)));
            %     sim_plv_roi(:,:,i) = sim_plv_roi(:,:,i) - diag(diag(sim_plv_roi(:,:,i)));
            % end
            % 平均不同 trial 的 PLV
            sim_plv = mean(sim_plv, 3);
            sim_plv_left = mean(sim_plv_left, 3);
            sim_plv_right = mean(sim_plv_right, 3);
            sim_plv_roi = mean(sim_plv_roi, 3);
            
            % 计算 PLV 平均值
            average_value(n-80) = mean(sim_plv(:));
            average_value_left(n-80) = mean(sim_plv_left(:));
            average_value_right(n-80) = mean(sim_plv_right(:));
            average_value_roi(n-80) = mean(sim_plv_roi(:));

            % 保存结果
            sim_plv_all(:,:,n-80) = sim_plv;
            sim_plv_left_all(:,:,n-80) = sim_plv_left;
            sim_plv_right_all(:,:,n-80) = sim_plv_right;
            sim_plv_roi_all(:,:,n-80) = sim_plv_roi;
        end

        % 找到最大平均值的位置
        [M, I] = max(average_value);

        % 平滑整体的 PLV 曲线
        plv_smdata = smooth(average_value, 0.2, 'lowess')';
        plv_smdata_left = smooth(average_value_left, 0.2, 'lowess')';
        plv_smdata_right = smooth(average_value_right, 0.2, 'lowess')';
        plv_smdata_roi = smooth(average_value_roi, 0.2, 'lowess')';

        % 保存整体 PLV 结果
        save('Data\sim_plv_all.mat','sim_plv_all','D','average_value')
        save('Data\sim_plv_left_all.mat','sim_plv_left_all','D','average_value_left')
        save('Data\sim_plv_right_all.mat','sim_plv_right_all','D','average_value_right')
        save('Data\sim_plv_roi_all.mat','sim_plv_roi_all','D','average_value_roi')

        % 保存平滑数据
        save('Data\phase_average_value.mat','average_value','plv_smdata','D','M','I', ...
            'average_value_left','plv_smdata_left','average_value_right','plv_smdata_right', ...
            'average_value_roi','plv_smdata_roi');

     case 15 % 不同阈值下相关性和相似性的变化
        load('Data\emp_plv.mat')
        a = [];phase_locking_value_origin_MCI = [];exc = [];C_origin = 0:0.1:90;
        n = 204;
        expName = ['exp' num2str(n, '%03d')];
        nTrial = 20;
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                excs_all(:,:,i) = excs;
                a(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));
                
            end
            exc = mean(excs_all,3);
            phase_locking_value_origin_MCI(n) = cal_fc_correlation(mean(a,3),emp_plv);
        [M_origin,I_origin]=max(phase_locking_value_origin_MCI);
        plv_fitdata_origin = smoothdata(phase_locking_value_origin_MCI,'sgolay',51);
        save('Data\phase_locking_value_origin_MCI.mat','phase_locking_value_origin_MCI','plv_fitdata_origin','C_origin','M_origin','I_origin')
        % save('Data/exc.mat','exc')
    case 16 % 模拟点处的锁相值功能连接矩阵
        n=204;
        load('Data\phase_locking_value_origin_MCI.mat')
        sim_plv = [];
        DD=C_origin(I_origin);C_origin=[];C_origin=DD;
        expName = ['exp' num2str(n, '%03d')];
        nTrial = 20;
        for i = 1: nTrial
            loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
            load(loadfilename)
            sim_plv(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));
        end 
        sim_plv = mean(sim_plv,3);
        average_value_origin = mean(sim_plv(:));
        save('Data\sim_plv_origin_MCI.mat','sim_plv','C_origin','average_value_origin')
    case 17 % 熵
        b = [];
        pe_values = [];
        D = 80:1:120; % TMS刺激强度范围
        nTrial = 50; % 实验轮次数

        for n = 81 : (80 + length(D))
            expName = ['exp' num2str(n, '%03d')];
            b = zeros(nTrial, 68); % 68个脑区

            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename, 'excs'); % 确保正确加载变量 excs

                for region = 1:size(excs, 2)
                    signal_no = excs(1001:2000, region);
                    sorted_signal_no = sort(signal_no);
                    signal_pdf_no = histcounts(sorted_signal_no, 'Normalization', 'probability');
                    signal_pdf_no(signal_pdf_no == 0) = [];
                    b(i, region) = -sum(signal_pdf_no .* log2(signal_pdf_no));
                end
            end

            pe_values(n-80, :) = mean(b, 1); % 计算所有实验轮次的平均熵
        end

        save('Data\pe_values.mat','pe_values','D')
end
