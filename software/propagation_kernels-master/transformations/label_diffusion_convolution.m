% LABEL_DIFFUSION_CONVOLUTION simple isotopic diffusion transformation for 
% grid graphs. 
%
% This function performs a simple label diffusion transformation
% for use in propagation_kernel_grid.m. Here the features are
% discrete distributions and these distributions are propagated
% along the edges of a grid graph.
%
% For use in propagation_kernel_grid.m, a closure would be created
% containing the filter matrix B, e.g.:
%
%transformation = @(A) label_diffusion_convolution(A, B);
%
% Usage:
%
% A = label_diffusion_convolution(A, B)
%
% Inputs:
%
% A: a cell array of N feature matrices of sizes (n x m x d) for grids of 
% size (n x m). n and m can be different for each grid graph. Diffusion is 
% performed on each dimension d of each grid independently.
%
% B: filter matrix.
%
% Outputs:
%
% A: a cell array of N updated feature matrices of sizes (n x m x d).
%
% See also PROPAGATION_KERNEL_GRID, MAKE_B.
% 
% Based on the implementation of calculate_hashes by
% (c) Roman Garnett, 2012--2014.
%
% Copyright (c) Marion Neumann, 2014.

function A = label_diffusion_convolution(A, B)
  
    if ~iscell(A)
        A = {A};
    end
        
    num_images = size(A, 1);
    for i=1:num_images

        im = A{i};
        for j = 1:size(im,3)
            im(:, :, j) = conv2(im(:, :, j), B, 'same');     % isotropic diffusion
        end
        % renormalize (to keep probabilities at border)
        A{i} = bsxfun(@times, im, 1 ./ sum(im, 3));
    end
end