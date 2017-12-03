% for processing ptc, protein datasets into the same format as Weisfeiler-Lehman graph kernel Toolbox

dataset = strvcat('ptc', 'proteins');
for ith_data = 1: size(dataset, 1)
    data = dataset(ith_data, :)
    datapath = strcat('../data/', data, '/');
    i = 0;
    Y = [];
    while 1
        i = i + 1;
        onegraph = strcat(datapath, num2str(i), '.mat');
        if exist(onegraph, 'file')~=2
            break
        end
        load(onegraph);
        G(i).am = A;
        G(i).nl.values = x;
        G(i).al = {};
        for j = 1:size(A, 1)
            tmp = A(j, :);
            tmp = find(tmp);
            G(i).al{j, 1} = tmp;
        end
        Y = [Y;y];
    end
    eval(strcat(data, '=G;'));
    eval(strcat('l', data, '=Y;'));
    save(strcat('../data/raw_data/', data, '.mat'), strcat(data), strcat('l', data));


end
    

