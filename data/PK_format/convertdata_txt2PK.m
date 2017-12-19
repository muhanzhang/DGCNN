% %
% Converter for graph datasets from:
%
%                  http://graphkernels.cs.tu-dortmund.de
%
% Parses .txt files to matlab arrays.
%
% Marion Neumann (m.neumann@wustl.edu)
% %

datasets = strvcat('COLLAB', 'IMDB-BINARY', 'IMDB-MULTI');
for ith_data = 1: size(datasets, 1)
    dataset = strcat(datasets(ith_data, :));
    path2data = ['../raw_data/' dataset '/'];    % path to folder containing data
    
    % graph labels
    graph_labels = dlmread([path2data dataset '_graph_labels.txt']);
    num_graphs = size(graph_labels,1);
    
    graph_ind = dlmread([path2data dataset '_graph_indicator.txt']);
    num_nodes = size(graph_ind,1);
    save(strcat(dataset, '_mat.mat'), 'graph_labels', 'graph_ind');
    
    % node labels
    try
        labels = dlmread([path2data dataset '_node_labels.txt']);
        save(strcat(dataset, '_mat.mat'), 'labels', '-append');
        if size(labels,1) ~= num_nodes
            fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_labels.txt']);
        end
    catch
        disp('No node labels for this dataset.')
    end
    
    % node attributes
    try
        attributes = dlmread([path2data dataset '_node_attributes.txt']);
        save(strcat(dataset, '_mat.mat'), 'attributes', '-append');
        if size(attributes,1) ~= num_nodes
            fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_attributes.txt']);
        end
    catch
        disp('No node attributes for this dataset.')
    end
    
    % edges, adjacency matrix
    edges = dlmread([path2data dataset '_A.txt']);
    num_edges = size(edges,1);
    
    % edge attributes (e.g. edge weights etc.)
    try
        edge_attr = dlmread([path2data dataset '_edge_attributes.txt']);
        if size(edge_attr,1) ~= num_edges
            fprintf('ERROR: Wrong number of edges in %s!\n', [dataset '_edge_attributes.txt']);
        end
        if size(edge_attr,2)>1
            fprintf('CAUTION: there are more than one edge attributes in %s!\n', [dataset '_edge_attributes.txt']);
            fprintf('CAUTION: only the first one is used in adjacency matrix.\n');
        end
    catch
        edge_attr = ones(num_edges,1);
        disp('No edge attributes for this dataset.')
    end
    
    A = spones(sparse(edges(:,1), edges(:,2), edge_attr(:,1), num_nodes, num_nodes));
    
    save(strcat(dataset, '_mat.mat'), 'A', '-append');
end
