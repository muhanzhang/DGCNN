% LABEL_DIFFUSION simple label diffusion transformation.
%
% This function performs a simple label diffusion transformation
% for use in propagation_kernel.m. Here the features are
% discrete distributions and these distributions are propagated
% along the edges of a graph.
%
% For use in propagation_kernel.m, a closure would be created
% containing the adjacency matrix A, e.g.:
%
%   transformation = @(features) label_diffusion(features, A);
%
% Usage:
%
%   features = label_diffusion(features, A)
%
% Inputs:
%
%   features: an (n x k) matrix of features, where n is the number of
%             nodes in all graphs and k is the dimension of the
%             features.
%          A: an (n x n) block diagonal matrix containing the
%             adjacency matrices for all graphs.
%
% Outputs:
%
%   features: an (n x k) matrix containing the updated features.
%
% See also PROPAGATION_KERNEL, LABEL_PROPAGATION.

% Copyright (c) Roman Garnett, 2012--2014.

function features = label_diffusion(features, A)

  features = A * features;

end