% For transforming the sparse adjacency matrices in 'DD' and 'NCI1' to dense
% in order to be read by torch.matio
% Note torch.matio does not support dense .mat

dataset = strvcat('DD', 'NCI1');
for ith_data = 1: size(dataset, 1)
    clear save_struct
    data = strcat(dataset(ith_data, :))
    dataname = strcat('../data/raw_data/', data, '.mat');
    load(dataname);
    X = eval(data);
    for i = 1: size(X, 2)
        am = full(X(i).am);
        l = X(i).nl.values;
        X(i).am = am;
    end
    save_struct.(data) = X;
    save(dataname, '-struct', 'save_struct', '-append');
end

