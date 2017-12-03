% For transforming adjacency matrices to random orders,
% need dataset saved as "name_random.mat".

dataset = strvcat('MUTAG_random');
for ith_data = 1: size(dataset, 1)
    clear save_struct
    data = strcat(dataset(ith_data, :))
    dataname = strcat('../data/raw_data/', data, '.mat');
    load(dataname);
    X = eval(data(1:end-7));
    for i = 1: size(X, 2)
        am = full(X(i).am);
        l = X(i).nl.values;
        %[~, ~, am, l] = palette_wl(am, l); % transform to wl vertex order
        perm = randperm(length(l));
        X(i).am = am(perm, perm);
        X(i).nl.values = l(perm);
    end
    save_struct.(data(1:end-7)) = X;
    save(dataname, '-struct', 'save_struct', '-append');
end

