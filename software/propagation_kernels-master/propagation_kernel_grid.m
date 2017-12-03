% PROPAGATION_KERNEL_GRID calculates the propagation kernel between data
% organized in grids.
%
% This function contains an implementation of the propagation kernel
% for grid graphs. Propagation kernels for general graphs are described in:
%
% Neumann, M., Patricia, N., Garnett, R., and Kersting, K. Efficient
% Graph Kernels by Randomization. (2012). Machine Learning And
% Knowledge Discovery in Databases: European Conference, (ECML/PKDD
% 2012), pp. 378-392.
%
% This implementation supports (approximately) preserving any of the
% following distances between the feature vectors:
%
% - \ell^1 or \ell^2 for arbitrary feature vectors
% - the total variation or Hellinger distance for
% distribution-valued features
%
% Depending on the chosen distance, the input features will be
% transformed appropriately.
%
% This implementation also supports arbitrary transformations to be
% used. See the transformations/ directory for example implementations
% (isotropic diffusion). Transformations must satisfy the following very
% general interface:
%
% A = transformation(A);
%
% Where the input is a cell array of N feature matrices of size (n x m x d). 
% One feature matrix per grid graph. (n x m) are the grid dimensions and d 
% is the feature dimension. The transformation function computes a new set 
% of features given an old set.
%
% Usage:
%
% K = propagation_kernel_grid(A, transformation, num_iterations, varargin)
%
% Inputs:
%
% A: a cell array of N feature matrices of sizes (n x m x d) to hash. n and 
% m can be different for each grid graph. If either the total variation or 
% Hellinger distance is chosen, the feature matrices should sum to 1 in 
% their 3rd dimension.
%
% transformation: a function handle to a feature transformation
% function satisfying the above-described API.
%
% num_iterations: the number of iterations to use for the kernel
% computation.
%
% Optional inputs (specified as name/value pairs):
%
% 'w': the bin width to use during the hashing
% computation (default: 1e-4)
% 
% 'distance': a string indicating the distance to approximately
% preserve; the following values are supported:
% 'l1': \ell^1
% 'l2': \ell^2
% 'tv': total variation distance (equivalent to \ell^1)
% 'hellinger': Hellinger distance
%
% The input is not case sensitive. See
% calculate_hashes for more information.
% (default: 'l1').
%
% 'base_kernel': a function handle to the base kernel to use. The
% kernel will be called as:
%
% K = base_kernel(counts),
%
% where counts is a (N x k) matrix, where m is the number of graphs and k 
% is the number of unique hashes during this step of the computation.
% counts(i, j) contains the number of times hash j occurs in graph i. 
% The default base kernel is the linear one:
%
% @(counts) (counts * counts');
%
% Outputs:
%
% K: an (N x N) matrix containing the computed propagation kernel.
%
% See also CALCULATE_HASHES_GRID, LABEL_DIFFUSION_CONVOLUTION.
% 
% 
% Based on the implementation of propagation_kernel by
% (c) Roman Garnett, 2012--2014.
%
% Copyright (c) Marion Neumann, 2014.

function K = propagation_kernel_grid(A, transformation, ...
                                                num_iterations, varargin)
  verbose = false;
  
  % parse optional inputs                          
  options = inputParser;
  
  % which distance to use
  options.addOptional('distance', 'l1', ...
                      @(x) ismember(lower(x), {'l1', 'l2', 'tv', 'hellinger'}));
  % width for hashing                
  options.addOptional('w', 1e-4, ...
                      @(x) (isscalar(x) && (x > 0)));
  % base kernel for counts vector, default is linear kernel                
  options.addOptional('base_kernel', ...
                      @(counts) (counts * counts'), ...
                      @(x) (isa(x, 'function_handle')));
  options.addOptional('num_steps', 1, ...
                      @(x) (isscalar(x) && (x > 0)));   % do num_stps feature updates per kernel contribution

  options.parse(varargin{:});
  options = options.Results;

  num_graphs = size(A,1);
  K = zeros(num_graphs);    % initialize output
  iteration = 0;
  while (true)

    % hashing
    if (verbose); fprintf('...computing hashvalues\n'); end
    hash_labels = calculate_hashes_grid(A, options.distance, options.w);
    
    % GET set of existing hash labels
    sth = cellfun(@(im) (unique(im(:))), hash_labels,'UniformOutput',false);
    label_set = unique(cell2mat(sth));
    min_label = min(label_set);
    num_labels = numel(label_set);
    label_set = label_set-min_label+1; 

    if num_labels > 200000
            fprintf('skipping kernel contribution at iteration %d (num hashlabels too large = %d)\n',...
                    iteration, num_labels);
    else
 
        % aggregate counts on graphs
        if (verbose); fprintf('...computing kernel contribution\n'); end        

        add_contrib = true;
        counts = zeros(num_graphs,num_labels);
        for i=1:num_graphs     
            hash_labels{i} = hash_labels{i}-min_label+1;
            try
                curr_counts = accumarray(hash_labels{i}(:), 1);
            catch
                add_contrib=false;
                warning('skipping kernel contribution at iteration %d (num hashlabels = %d)\n', iteration, num_labels);
                break
            end
            idx = ismember(label_set,find(curr_counts));
            counts(i,idx) = curr_counts(find(curr_counts));
        end

        % contribution specified by base kernel on count vectors
        if add_contrib
            K = K + options.base_kernel(counts);
        end
    end
    % avoid unnecessary transformation on last iteration
    if (iteration == num_iterations)
      break;
    end

    % apply transformation to features for next step
    if (verbose); fprintf('...label update\n'); end
    for step=1:options.num_steps
        A = transformation(A);
    end
    iteration = iteration + 1;
  end
end
