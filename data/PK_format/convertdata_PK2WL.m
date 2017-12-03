% For converting PK graph format into WL graph format.
clear all;
dataset = strvcat('COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY');
for ith_data = 1: size(dataset, 1)
    clear save_struct
    clear save_struct2
    clear labels
    clear attributes
    data = strcat(dataset(ith_data, :))
    load(strcat(data, '_mat.mat'));
    X = {};
    for i = 1: length(graph_labels)
        am = A(graph_ind == i, graph_ind == i);
        al = cellfun(@(x) find(x),num2cell(am, 2), 'un', 0);
        nl = {}; % node label
        na = {}; % node attribute
        if exist('labels', 'var')
            nl.values = labels(graph_ind == i);
        end
        if exist('attributes', 'var')
            na.values = attributes(graph_ind == i, :);
        end
        X(i).am = full(am);
        X(i).nl = nl;
        X(i).na = na;
        X(i).al = al;
    end
    save_struct.(data) = X;
    save(['../raw_data/' data '.mat'], '-struct', 'save_struct');
    save_struct2.(['l' lower(data)]) = graph_labels;
    save(['../raw_data/' data '.mat'], '-struct', 'save_struct2', '-append');
end
