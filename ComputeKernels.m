% Usage: compute kernel matrices of some datasets using three
%        kernels WL, RW, SP, the computed kernel matrices are 
%        saved in data/kernels/XYZ.mat
%
% *author: Muhan Zhang, Washington University in St. Louis
clear all;
addpath(genpath('software/graphkernels'));

% specify the graph kernel, 1: WL, 2: GK, 3: LRW
for kernel = 1:3
    switch kernel
        case 1 % WL
            dataset = strvcat('MUTAG', 'ptc', 'NCI1', 'proteins', 'DD');
            original_nl = 1;  % if you want to use original node labels in WL
        case 2 % GK
            dataset = strvcat('MUTAG', 'ptc', 'NCI1', 'proteins', 'DD');
        case 3 % labeled-RW
            dataset = strvcat('MUTAG', 'ptc', 'proteins'); % for RW, the other two datasets take > 3 days
            RWlambda = ones(1, 3) * 1e-3
    end

    for ith_data = 1: size(dataset, 1)
        tic;
        data = dataset(ith_data, :)
        dataname = strcat('data/raw_data/', data, '.mat');
        load(dataname);
        destination = strcat('data/kernels/', data);
        mkdir(destination);
        X = eval(data);  % the graph X
        l = eval(lower(strcat('l', data)));  % the labels Y

        % WL subtree kernel
        switch kernel
            case 1 % WL
                [K, runtime] = WL(X, 5, original_nl);
                display('kernel computation time:')
                runtime
                save(strcat(destination, '/WL.mat'), 'K', 'l');
            case 2 % GK
                [K, runtime] = allkernel(X, 3);
                runtime
                save(strcat(destination, '/GK.mat'), 'K', 'l');
            case 3 % labeled random walk
                [K, runtime] = lRWkernel(X, RWlambda(ith_data), 1);
                runtime
                save(strcat(destination, '/LRW.mat'), 'K', 'l');
        end
        toc
    end
  
end
