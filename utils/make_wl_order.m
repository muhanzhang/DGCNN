% For transforming adjacency matrices to the orders imposed by WL,
% need dataset saved as "name_wl.mat".

dataset = strvcat('MUTAG_wl');
for ith_data = 1: size(dataset, 1)
    clear save_struct
    data = strcat(dataset(ith_data, :))
    dataname = strcat('../data/raw_data/', data, '.mat');
    load(dataname);
    X = eval(data(1:end-3));
    for i = 1: size(X, 2)
        am = full(X(i).am);
        l = X(i).nl.values;
        [~, ~, am, l] = palette_wl(am, l); % transform to wl vertex order
        X(i).am = am;
        X(i).nl.values = l;
    end
    save_struct.(data(1:end-3)) = X;
    save(dataname, '-struct', 'save_struct', '-append');
end

