function varargout = setfiles(varargin)

dirname = 'Data\';

for i = 1 : length(varargin)

    name = varargin{i};

    switch lower(name)
        
        case 'weight'
            target = [dirname 'Cij.mat'];
        case 'distance'
            target = [dirname 'Dij.mat'];
        case 'fc'
            target = [dirname 'fc.mat'];
        otherwise
            target = [];
    
    end
    
    varargout{i} = target;

end
