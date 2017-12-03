
% Copyright (c) Roman Garnett, 2012--2014.

function features = gmm_propagation(A, mus, sigmas, x)

  persistent weights;

  % reset if no input arguments given
  if (nargin == 0)
    weights = [];
    return;
  end

  % initialize weights if required
  if (isempty(weights))
    num_nodes = size(A, 1);
    weights = speye(num_nodes);

  % otherwise, diffuse weights
  else
    weights = A * weights;
  end

  % evaluate GMM mixture PDFs given weights
  features = evaluate_pdfs(mus, sigmas, x, weights);

end