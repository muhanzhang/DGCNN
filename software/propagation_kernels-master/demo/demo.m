rng('default');
addpath(genpath('..'));
% this is to illustrate the use of PARTIALLY LABELED graphs
% (missing node information) 
% DO USE ALL NODE LABELS IF AVAILABLE
num_train      = 1e3;    % number of nodes to use for the training set

% propagation kernel parameters
num_iterations = 20;     % number of iterations

% hashing
w              = 1e-2;          % bin width
distance       = 'hellinger';   % distance to approximately preserve

load('mutag');

num_nodes   = size(A, 1);
num_classes = max(labels);

% choose a random training set
train_ind       = randperm(num_nodes, num_train);
observed_labels = labels(train_ind);

% generate initial feature vectors for propagation kernel. here we
% use a delta distribution on the known classes for training nodes
% and a uniform class distribution on the others.

initial_features = repmat(ones(1, num_classes) / num_classes, [num_nodes, 1]);
initial_features(train_ind, :) = ...
    accumarray([(1:num_train)', observed_labels], 1, ...
               [num_train, num_classes]);

% create a function handle to a feature transformation. here we will
% use label propagation given our selected training set.
transformation = @(features) ...
    label_propagation(features, A, train_ind, observed_labels);

% calculate the graph kernel using the default (linear) base kernel
K = propagation_kernel(initial_features, graph_ind, transformation, ...
                       num_iterations, ...
                       'distance', distance, ...
                       'w',        w);

figure;
set(gcf, 'color', 'white');
subplot(1, 2, 1);
imagesc(K(:,:,end));
axis('square');
title('linear kernel');
colorbar;

% try with label diffusion instead
transformation = @(features) label_diffusion(features, A);

% replace the default linear base kernel with an RBF kernel
length_scale = 3;
base_kernel = @(counts) ...
              exp(-(squareform(pdist(counts)).^2 / (2 * length_scale^2)));

% calculate the graph kernel again using the new parameters
K = propagation_kernel(initial_features, graph_ind, transformation, ...
                       num_iterations, ...
                       'distance',    distance, ...
                       'w',           w, ...
                       'base_kernel', base_kernel);

subplot(1, 2, 2);
imagesc(K(:,:,end));
axis('square');
title('RBF kernel');
colorbar;
