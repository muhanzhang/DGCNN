% CALCULATE_HASHES compute locality sensitive hashes.
%
% This function computes a locality sensitive hash for vectors in the
% manner described in:
%
%   Neumann, M., Patricia, N., Garnett, R., and Kersting, K. Efficient
%   Graph Kernels by Randomization. (2012). Machine Learning And
%   Knowledge Discovery in Databases: European Conference, (ECML/PKDD
%   2012), pp. 378-392.
%
% This implementation supports (approximately) preserving any of the
% following distances:
%
%   - \ell^1 or \ell^2 for arbitrary feature vectors
%   - the total variation or Hellinger distance for
%     distribution-valued features
%
% Depending on the chosen distance, the input features will be
% transformed appropriately.
%
% Usage:
%
%   labels = calculate_hashes(features, distance, w)
%
% Inputs:
%
%    features: an (n x d) matrix of feature vectors to hash. If either
%              the total variation or Hellinger distances is chosen,
%              each row of this matrix should sum to 1.
%    distance: a string indicating the distance; the following
%              values are supported:
%
%                'l1': \ell^1
%                'l2': \ell^2
%                'tv': total variation distance (equivalent to \ell^1)
%         'hellinger': Hellinger distance
%
%              The input is not case sensitive.
%
%           w: the bin width
%
% Outputs:
%
%   labels: an (n x 1) vector containing the hashed features. The
%           hashes are guaranteed to be positive integers in a
%           contiguous range starting from 1.
%
% See also PROPAGATION_KERNEL.

% Copyright (c) Roman Garnett, 2012--2014.

function labels = calculate_hashes(features, distance, w)

  % determine path to take depending on chosen distance
  use_cauchy = (strcmpi(distance, 'l1') || strcmpi(distance, 'tv'));
  take_sqrt  = strcmpi(distance, 'hellinger');

  [~, d] = size(features);

  if (take_sqrt)
    features = sqrt(features);
  end

  % generate random projection vector
  v = randn(d, 1);
  if (use_cauchy)
    % if X, Y are N(0, 1), then X / Y has a standard Cauchy distribution
    v = v ./ randn(d, 1);
  end

  % random offset
  b = w * rand;

  % compute hashes
  labels = floor((features * v + b) * (1 / w));

  % map to contiguous integers starting from 1
  [~, ~, labels] = unique(labels);

end