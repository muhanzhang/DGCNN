% MAKE_B creates a weighted symmetric filter matrix encoding the
% neighborhood for isotropic label diffusion. 
%
% Usage:
% 
% B = make_B(1,4);
% Returns a 3 x 3 symmetric filter matrix encoding a 4-neighborhood 
% corresponding to 4 equally spaced points on a 1 point radius of the 
% center point.   
%
% B = make_B(2,16);
% Returns a 5 x 5 symmetric filter matrix encoding 16 equally spaced points 
% on a 2 point radius of the center point.  
%
% Inputs:
%
% radius: distance from the center point used to determine filter matrix.
% num_points: number of neighbors used in the filter matrix.
%
% Output:
%
% B: filter matrix (approximately) encoding num_points equally spaced points 
% on the speccified radius from the center point. 
%
% Copyright (c) Roman Garnett, 2012--2014.

function B = make_B(radius, num_points)

  factor = 100;

  small_length = 2 * radius + 1;
  big_length = factor * small_length;

  big_B = zeros(big_length);

  offset = big_length / 2;
  side = factor / 2;

  for i = 1:num_points
    angle = (i - 1) / num_points * 2 * pi;

    x = round(offset + factor * radius * cos(angle));
    y = round(offset + factor * radius * sin(angle));

    x_lower = max(x - side, 1);
    x_upper = min(x + side, big_length);
    y_lower = max(y - side, 1);
    y_upper = min(y + side, big_length);

    big_B(x_lower:x_upper, y_lower:y_upper) = ...
        big_B(x_lower:x_upper, y_lower:y_upper) + 1;
  end

  B = zeros(small_length, small_length);

  for i = 1:small_length
    for j = 1:small_length
      x_range = (1 + (i - 1) * factor):(i * factor);
      y_range = (1 + (j - 1) * factor):(j * factor);

      B(i, j) = sum(sum(big_B(x_range, y_range)));
    end
  end

  B = B + rot90(B, 1) + rot90(B, 2) + rot90(B, 3);
  B = B / sum(B(:));

  B = round(B * 100) / 100;
end