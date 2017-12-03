% LABEL_PROPAGATION label propagation transformation.
%
% This function performs a label propagation transformation for use in
% propagation_kernel.m. Here the features are discrete distributions
% and these distributions are propagated along the edges of a
% graph. The propagation is performed given a training set.
%
% For use in propagation_kernel.m, a closure would be created
% containing the adjacency matrix A, e.g.:
%
%   transformation = @(features) ...
%                    label_propagation(features, A, train_ind, observed_labels)
%
% Usage:
%
%   features = label_propagation(features, A, train_ind, observed_labels)
%
% Inputs:
%
%          features: an (n x k) matrix of features, where n is the
%                    number of nodes in all graphs and k is the
%                    number of classes.
%                 A: an (n x n) block diagonal matrix containing the
%                    adjacency matrices for all graphs.
%         train_ind: an (m x 1) vector of indices into A/features
%                    indicating the nodes with known labels
%   observed_labels: an (m x 1) vector of associated observed
%                    labels from 1 to k
%
% Outputs:
%
%   features: an (n x k) matrix containing the updated features.
%
% See also PROPAGATION_KERNEL, LABEL_DIFFUSION.

% Copyright (c) Roman Garnett, 2012--2014.

function features = label_propagation(features, A, train_ind, ...
          observed_labels)

  num_train   = numel(observed_labels);
  num_classes = size(features, 2);

  features = A * features;

  % "push back" training labels
  features(train_ind, :) = ...
      accumarray([(1:num_train)', observed_labels], 1, ...
                 [num_train, num_classes]);

end