% %
% Setup default values and run PK computation. Note that the kernel will
% perform BETTER if you learn the parameters via cross validation instead of
% using these defaults. 
% 
% Marion Neumann (m.neumann@wustl.edu)
% % 

% propagation kernel parameter
num_iterations = 3;     % number of iterations (sth small 2 or 3)

% hashing parameters
w              = 1e-5;  % bin width
distance       = 'tv';  % distance to approximately preserve

% load you dataset HERE
load('mutag_mat');      

num_nodes   = size(A, 1);
num_classes = max(labels);

initial_label_distributions = accumarray([(1:num_nodes)', labels], 1, [num_nodes, num_classes]);

% create a function handle to a feature transformation. Here we will
% use label diffustion as we have fully labeled graphs.
transformation = @(features) label_diffusion(features, A);


% calculate the graph kernel using the default (linear) base kernel
K = propagation_kernel(initial_label_distributions, graph_ind, transformation, ...
                       num_iterations, ...
                       'distance', distance, ...
                       'w',        w);


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
