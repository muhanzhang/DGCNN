
% Copyright (c) Roman Garnett, 2012--2014.

function pdfs = evaluate_pdfs(mus, Ks, x, weights)

  persistent individual_pdfs;

  % reset if no arguments given
  if (nargin == 0)
    individual_pdfs = [];
    return;
  end

  % precompute N(x; mu, K) for every x
  if (isempty(individual_pdfs))
    num_points = size(x, 1);
    num_nodes  = size(mus, 1);

    individual_pdfs = zeros(num_nodes, num_points);
    for i = 1:num_nodes
        individual_pdfs(i, :) = mvnpdf(x, mus(i, :), Ks(:, :, i))';
    end
  end

  pdfs = weights * individual_pdfs;

end