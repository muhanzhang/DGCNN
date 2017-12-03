% Main program for comparison with graph kernels using the same train/val/test splits
% *author: Muhan Zhang, Washington University in St. Louis

clear all
seed_start = 100;
repeat_times = 10;
cv = 10;
addpath(genpath('software/graphkernels'));

kernels = strvcat('WL', 'GK', 'LRW', 'PK');
dataset = strvcat('MUTAG', 'DD', 'ptc', 'NCI1', 'proteins');

Res = zeros(size(kernels, 1), size(dataset, 1), repeat_times);
Time = zeros(size(kernels, 1), size(dataset, 1), repeat_times);

for seed = seed_start: seed_start + repeat_times - 1 
    rand('seed', seed);
    t = seed + 1 - seed_start;

    for ith_data = 1: size(dataset, 1)
        data = dataset(ith_data, :)
        clear r
        for ith_kernel = 1: size(kernels, 1)
            tic;
            kernel = kernels(ith_kernel, :)
            kernelname = strcat('data/kernels/', data, '/', kernel, '.mat');
            if ~exist(kernelname, 'file')
                continue
            else
                load(kernelname); % get K and l
            end
            if ~iscell(K)  % for GK and LRW, K is only one-fold, make it to a cell
                tmp{1, 1} = K;
                K = tmp;
            end
            n = length(l);
            p80 = ceil(n * (1-2/cv));
            p90 = ceil(n * (1-1/cv));
            fs = n - p90; % fold size
            if ~exist('r')
                r = randperm(n);
                mkdir('data/shuffle');
                for k = 1: cv
                    r_current = r([k*fs+1:n,1:(k-1)*fs,(k-1)*fs+1:k*fs]);
                    % save the shuffle indices r for running DGCNN using same splits
                    %save(strcat('data/shuffle/', data, num2str(t), '_', num2str(k), '.mat'), 'r_current');  
                end
            end
            res = runsvm(K, l, r);
            Res(ith_kernel, ith_data, t) = res.mean_acc;
            Time(ith_kernel, ith_data, t) = toc;
        end
    end
end
        
AvgRes = mean(Res, 3);
StdRes = std(Res, 0, 3);
SumTime = sum(Time, 3);

display('Average results:')
AvgRes
StdRes
SumTime
