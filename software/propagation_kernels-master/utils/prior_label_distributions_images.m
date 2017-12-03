% PRIOR_LABEL_DISTRIBUTIONS_IMAGES quantizes rgb or intensity images 
% organized in (n x m) grids and returns delta distributions among the 
% quantization values per pixel. 
%
% Usage:
%
% inital_A = prior_label_distributions_images(A,num_col);
%
% Inputs:
%
% A: a cell array of N image matrices of sizes (n x m x 3) for rgb images 
% and (n x m) for intensity images. n and m can be different for each image 
% grid. 
%
% num_col: number of quantization levels.
%
% Outputs:
%
% inital_A: a cell array of N inital label distribution matrices of sizes 
% (n x m x num_col) to use in propagation_kernel_grid.
%
% See also PROPAGATION_KERNEL_GRID, LABEL_DIFFUSION_CONVOLUTION.
% 
% 
% Copyright (c) Marion Neumann, 2014.

function A_distributions = prior_label_distributions_images(A,feat_dim)

    num_images = size(A, 1);
    A_distributions = cell(size(A));

    for i=1:num_images
        im_hist = zeros(size(A{i}, 1),size(A{i}, 2), feat_dim);
        
        im = A{i};
        if max(max(im))~=1
            im = im - min(min(im));
            im = im/max(max(im));
        end
        
        if size(A, 3) == 3
            im = uint16(rgb2gray(im)*(feat_dim-1));
        else
            im = uint16(im*(feat_dim-1));
        end
        
        for j=1:size(A{i}, 1)
            for k=1:size(A{i}, 2)
                im_hist(j,k,im(j,k)+1) = 1;
            end
        end
      
        A_distributions{i}=double(im_hist);
    end
  
end