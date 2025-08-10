function simulate_model(expName)

FUNCTION = 'simulate_model';

Param = expfile_load(['Expfile\' expName '.txt']);
Cij = load(setfiles('weight'));
% Dij = load(setfiles('distance'));
Param.Cij = Cij.Cij;
% Param.Dij = Dij.Dij;

for n = 1 : 30
    savefilename = sprintf('Results\\%s\\sim_trial%02d.mat',expName, n);
    tic
    if exist(savefilename, 'file') == 0
        fprintf('%s[%s] simulating trial %02d \n', FUNCTION, expName, n);
        excs = simulate_wendModel(Param);
        if exist(fileparts(savefilename), 'dir') == 0
            mkdir(fileparts(savefilename));
        end
        save(savefilename, 'excs')
    end
    time = toc;
    fprintf('%s[%s] save %s. \n',FUNCTION, expName, savefilename);
    fprintf('%s[%s] elapsed time %.2f sec. \n',FUNCTION, expName, time);
end
end

