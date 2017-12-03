% Usage: to compute the PK kernel matrices
% 
% Setup default values and run PK computation. Note that the kernel will
% perform BETTER if you learn the parameters via cross validation instead of
% using these defaults. 
% 
% Marion Neumann (m.neumann@wustl.edu)
% % 

addpath(genpath('software/propagation_kernels-master'));

% propagation kernel parameter
num_iterations = 5;     % number of iterations

% hashing parameters
w              = 1e-3;  % bin width
distance       = 'tv';  % distance to approximately preserve


% load you dataset HERE
dataset = strvcat('MUTAG', 'ptc', 'NCI1', 'proteins', 'DD');
for ith_data = 1: size(dataset, 1)
    clear labels
    clear l
    clear graph_labels
    tic
    data = dataset(ith_data, :)
    load(strcat('data/PK_format/', data, '_mat.mat'));

    if ~exist('labels', 'var') 
        labels = full(sum(A, 2));
        labels(labels==0) = 1;
    end
    if ~exist('l', 'var')
        if exist('graph_labels', 'var')
            l = graph_labels;
        end
    end

    num_nodes   = size(A, 1);   % number of nodes
    num_classes = max(labels);  % number of node label classes
    % row-normalize adjacecny matrix A
    row_sum = sum(A, 2);
    row_sum(row_sum==0)=1;  % avoid dividing by zero => disconnected nodes
    A = bsxfun(@times, A, 1 ./ row_sum);
    

    initial_label_distributions = accumarray([(1:num_nodes)', labels], 1, [num_nodes, num_classes]);

    % create a function handle to a feature transformation. Here we will
    % use label diffustion as we have fully labeled graphs.
    transformation = @(features) label_diffusion(features, A);


    % calculate the graph kernel using the default (linear) base kernel
    K = propagation_kernel(initial_label_distributions, graph_ind, transformation, ...
                           num_iterations, ...
                           'distance', distance, ...
                           'w',        w);

    Ks = {};
    for i = 1:size(K, 3)
        Ks{1, i} = K(:, :, i);
    end
    K = Ks;
    

    save(strcat('data/kernels/', data, '/PK.mat'), 'K', 'l');


    % % If you want to set the BASE KERNEL to an RBF kernel instead of the default 
    % % linear base kernel, use the following.
    % length_scale = 3;
    % base_kernel = @(counts) ...
    %               exp(-(squareform(pdist(counts)).^2 / (2 * length_scale^2)));
    % 
    %           % calculate the graph kernel again using the new parameters
    % K = propagation_kernel(initial_label_distributions, graph_ind, transformation, ...
    %                        num_iterations, ...
    %                        'distance',    distance, ...
    %                        'w',           w, ...
    %                        'base_kernel', base_kernel);
    toc
end
