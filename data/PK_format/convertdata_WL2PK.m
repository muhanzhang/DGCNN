% %
% Converter for graph datasets MATLAB struct format (used in WL implementataion)  
% to matrix/array format (used in propagation kernels PK implementation). 
%                  
%
% Parses graphs in matlab array of structs to sparse adjacency matrix, 
% graph indicator vector and (node) label vector.
% 
% Marion Neumann (m.neumann@wustl.edu)
% %  
%
% USAGE: replace ALL instances of DS with the name of the dataset 
% (all capital letters, e.g. NCI1) 

dataset = strvcat('MUTAG', 'ptc', 'NCI1', 'proteins', 'DD');
for ith_data = 1: size(dataset, 1)
    data = strcat(dataset(ith_data, :))


load(strcat('../raw_data/', data, '.mat'))
X = eval(data);
l = eval(lower(strcat('l', data)));

num_graphs = size(X,2);

%graph_labels = lds;  %<---------- you will have to use this if you want the graph labels
%if size(labels,1)==1
%    labels = labels';
%end

labels = [];        % node_labels
graph_ind = [];     % graph indicator vector
figure;
%for i= 1:num_graphs
for i = 1:0
       
    if i == 1
        A = sparse(X(i).am);
    else
        A = blkdiag(A, sparse(X(i).am));
    end    


    spy(X(i).am), title(num2str(i))
    pause(0.5)

    labels = [labels; X(i).nl.values];
    graph_ind = [graph_ind; i*ones(size(X(i).nl.values,1),1)];
end
save(strcat(data, '_mat.mat'), 'A', 'labels', 'graph_ind', 'l');

%save(strcat(data, '_mat.mat'), 'l', '-append');

end
