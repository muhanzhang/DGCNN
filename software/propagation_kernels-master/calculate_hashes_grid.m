
% CALCULATE_HASHES_GRID compute locality sensitive hashes.
% This function computes a locality sensitive hash for features organized 
% in a cell array of N matrices of size (n x m x d). The hashing procedure 
% is described in:
%
% Neumann, M., Patricia, N., Garnett, R., and Kersting, K. Efficient
% Graph Kernels by Randomization. (2012). Machine Learning And
% Knowledge Discovery in Databases: European Conference, (ECML/PKDD
% 2012), pp. 378-392.
%
% This implementation supports (approximately) preserving any of the
% following distances:
%
% - \ell^1 or \ell^2 for arbitrary feature vectors
% - the total variation or Hellinger distance for distribution-valued 
% features
%
% Depending on the chosen distance, the input features will be
% transformed appropriately.
%
% Usage:
%
% labels = calculate_hashes_grid(A, distance, w)
%
% Inputs:
%
% A: a cell array of N feature matrices of sizes (n x m x d) to hash. n and 
% m can be different for each grid graph. If either the total variation or 
% Hellinger distance is chosen, the feature matrices should sum to 1 in 
% their 3rd dimension.
%
% distance: a string indicating the distance; the following
% values are supported:
% 'l1': \ell^1
% 'l2': \ell^2
% 'tv': total variation distance (equivalent to \ell^1)
% 'hellinger': Hellinger distance
% The input is not case sensitive.
%
% w: the bin width
%
% Outputs:
%
% labels: a cell array of N matrices of size (m x n) containing the hashed 
% features. 
%
% See also PROPAGATION_KERNEL_GRID.
%
% Based on the implementation of calculate_hashes by
% (c) Roman Garnett, 2012--2014.
%
% Copyright (c) Marion Neumann, 2014.

function labels = calculate_hashes_grid(A, distance, w)
    
  % A:  cell array of d-dim matrices  
  labels = cell(size(A));
  
  % determine path to take depending on chosen distance
  use_cauchy = (strcmpi(distance, 'l1') || strcmpi(distance, 'tv'));
  take_sqrt  = strcmpi(distance, 'hellinger');

  if (take_sqrt)
    A = cellfun(@sqrt, A, 'UniformOutput', false);
  end

  % generate random projection vector 
  d = size(A{1},3);
  v = randn(d, 1);
  
  if (use_cauchy)
    % if X, Y are N(0, 1), then X / Y has a standard Cauchy distribution
    v = v ./ randn(d, 1);
  end

  % random offset
  b = w * rand;

  % compute hashes
  new_v = zeros(1,1,d);
  new_v(1,1,:) = v;
  for i=1:size(A,1)
      labels{i,1} = floor((sum(bsxfun(@times, A{i}, new_v), 3) + b) * (1/w));
  end
  
end
