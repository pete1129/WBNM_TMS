% -----------------------
% Calculate simulation data correlation analysis indicators
% -----------------------
%Case 1 Correlation of phase-locked values and smoothing filter curves for different global coupling coefficients (matrix-wise correlation defined as the correlation between lower triangular and upper triangular elements)
%Case 2 Baseline potentials of different brain regions with varying global coupling coefficients
%Case 3 Main frequencies of different brain regions with varying global coupling coefficients
%Case 4 Baseline potentials of different brain regions at simulation points
%Case 5 Time series, power spectral density, and main frequencies of different brain regions at simulation points
%Case 6 Phase-locked value functional connectivity matrix at simulation points
%Case 7 Calculation of Pearson correlation and phase-locked value correlation for different numbers of trials
%Case 8 Calculation of PLV graph theory analysis metrics
%Case 9 Changes in correlation and similarity under different thresholds
%Case 10 Entropy values of each brain region at simulation points
clc;clear;close all;
metric = 14;
fs = 1000;
nTrial = 50;
expnum = dir(['Expfile\' '*.txt']);

switch metric
    
    case 1 
        load('Data\emp_plv.mat')
        a = [];phase_locking_value = [];excs_all = [];D = 0.8:0.01:1.2;
        for n = 81 : (80 + length(expnum))
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
    case 2 
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
    
    case 3 
        c = [];dominant_rhythm = [];D=80:1:120;
        for n = 81 : (80 + length(expnum))
            expName = ['exp' num2str(n, '%03d')];
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                x = detrend(excs(1801:2000,:));
                N = length(x);
                nfft = 2^nextpow2(N);
                [per,f] = periodogram(x,[],nfft,fs);
                [M,I] = max(per);
                for j = 1 : N
                    c(i,j) =f(I(j));
                end
            end
            dominant_rhythm(n-80,:) = mean(c);    
        end
        save('Data\dominant_rhythm_800.mat','dominant_rhythm','D')
    
    case 4
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
    
    case 5 
        load('Data\correlation_coefficient.mat')
        load('Data\phase_locking_value.mat')
        c = [];In=[];sim_dominant_rhythm=[];sim_excs=[];sim_psd=[];sim_ns=[];D=80:1:120;
        % [M,In(1)] = max(correlation_coefficient);
        % [M,In(2)] = max(phase_locking_value);
        % DD=[];DD(1)=D(In(1));DD(2)=D(In(2));D=[];D=DD;
        for n = 81 : (80+length(D))
            expName = ['exp' num2str(In(n), '%03d')];
            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename)
                x = detrend(excs(1001:2001,:));
                N = length(x);
                nfft = 2^nextpow2(N);
                [per,f] = periodogram(x,[],nfft,fs);
                [M,I] = max(per);
                for j = 1 : N
                    c(i,j) =f(I(j));
                end
            end
            for k=1:N
                [~,nspectrum(:,k)] = nom_spectrum(excs(1001:2001,k),fs);
            end
            t=0:0.001:1.5;
            sim_dominant_rhythm(1,:) = mean(c);
            sim_excs(:,:,n-80) = excs;
            sim_psd(:,:,n-80) = per;
            sim_ns(:,:,n-80) = nspectrum;
            
        end
        save('Data\sim_excs.mat','sim_excs','D','t')
        save('Data\sim_psd.mat','sim_psd','D','f')
        save('Data\sim_ns.mat','sim_ns','D','f')
        save('Data\sim_dominant_rhythm.mat','sim_dominant_rhythm','D')
    
    case 6
        load('Data\phase_locking_value.mat')
        sim_plv = [];
        DD=D(I);D=[];D=DD;
        expName = ['exp' num2str(I+79, '%03d')];
        nTrial = 50;
        for i = 1: nTrial
            loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
            load(loadfilename)
            sim_plv(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));
        end
        sim_plv = mean(sim_plv,3);
        average_value = mean(sim_plv(:));
        save('Data\sim_plv.mat','sim_plv','D','average_value')
        % sim_plv = [];
        % sim_plv_left = [];
        % sim_plv_right = [];
        % sim_plv_roi = [];
        % D = 80:1:120;
        % nTrial = 50;
        % 
        % left_indices = 1:34;
        % right_indices = 35:N;
        % 
        % % ROI
        % roi_indices = [9,11,13,15,22,24,25]; % DMN
        % 
        % for n = 81 : (80 + length(D))
        %     expName = ['exp' num2str(n, '%03d')];
        % 
        %     for i = 1:nTrial
        %         loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat', expName, i);
        %         load(loadfilename) % excs
        %         sim_plv(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001,:)));
        %         sim_plv_left(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001, left_indices)));
        %         sim_plv_right(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001, right_indices)));
        %         sim_plv_roi(:,:,i) = nonfilter_cal_plv(detrend(excs(1001:2001, roi_indices)));
        %     end
        % 
        %     sim_plv = mean(sim_plv, 3);
        %     sim_plv_left = mean(sim_plv_left, 3);
        %     sim_plv_right = mean(sim_plv_right, 3);
        %     sim_plv_roi = mean(sim_plv_roi, 3);
        % 
        %     average_value(n-80) = mean(sim_plv(:));
        %     average_value_left(n-80) = mean(sim_plv_left(:));
        %     average_value_right(n-80) = mean(sim_plv_right(:));
        %     average_value_roi(n-80) = mean(sim_plv_roi(:));
        % 
        %     sim_plv_all(:,:,n-80) = sim_plv;
        %     sim_plv_left_all(:,:,n-80) = sim_plv_left;
        %     sim_plv_right_all(:,:,n-80) = sim_plv_right;
        %     sim_plv_roi_all(:,:,n-80) = sim_plv_roi;
        % end
        % [M, I] = max(average_value);
        % 
        % plv_smdata = smooth(average_value, 0.2, 'lowess')';
        % plv_smdata_left = smooth(average_value_left, 0.2, 'lowess')';
        % plv_smdata_right = smooth(average_value_right, 0.2, 'lowess')';
        % plv_smdata_roi = smooth(average_value_roi, 0.2, 'lowess')';
        % 
        % save('Data\sim_plv_all.mat','sim_plv_all','D','average_value')
        % save('Data\sim_plv_left_all.mat','sim_plv_left_all','D','average_value_left')
        % save('Data\sim_plv_right_all.mat','sim_plv_right_all','D','average_value_right')
        % save('Data\sim_plv_roi_all.mat','sim_plv_roi_all','D','average_value_roi')
        % 
        % 
        % save('Data\phase_average_value.mat','average_value','plv_smdata','D','M','I', ...
        %     'average_value_left','plv_smdata_left','average_value_right','plv_smdata_right', ...
        %     'average_value_roi','plv_smdata_roi');


    case 7 
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
    
    case 8
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
    
    case 9 
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
 
    case 10 % entropy
        b = [];
        pe_values = [];
        D = 80:1:120; 
        nTrial = 50; 

        for n = 81 : (80 + length(D))
            expName = ['exp' num2str(n, '%03d')];
            b = zeros(nTrial, N);

            for i = 1 : nTrial
                loadfilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName,i);
                load(loadfilename, 'excs'); 

                for region = 1:size(excs, 2)
                    signal_no = excs(1001:2000, region);
                    sorted_signal_no = sort(signal_no);
                    signal_pdf_no = histcounts(sorted_signal_no, 'Normalization', 'probability');
                    signal_pdf_no(signal_pdf_no == 0) = [];
                    b(i, region) = -sum(signal_pdf_no .* log2(signal_pdf_no));
                end
            end

            pe_values(n-80, :) = mean(b, 1); 
        end

        save('Data\pe_values.mat','pe_values','D')
end

