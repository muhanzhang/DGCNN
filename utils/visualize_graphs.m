%% Usage: graph visualization in MATLAB

rng('default');
datapath = '../data/raw_data/';
data = 'IMDBMULTI';
dataset = strcat(datapath, data, '.mat');
load(dataset);
G = eval(data);
Y = eval(strcat('l', lower(data)));
n = length(Y);
node_num = zeros(1, n);
for i = 1: n
    g = G(i);
    node_num(i) = size(g.am, 1);
end
std(node_num, 0, 2)
max(node_num)
min(node_num)
tmp = sort(node_num);
tmp(ceil(n * 0.6))




perm = randperm(n);

close all;

Counter = [];
nodeNum = [];
maxNodeDeg = 0;
for i = 1: n
    if mod(i, 25) == 0
        close all;
    end
    j = perm(i);
    a1 = G(j).am;
%     x1 = G(j).nl.values;
%     names0 = num2str(x1);
    nodeNum = [nodeNum, size(a1, 1)];
    maxNodeDeg = max(maxNodeDeg, max(sum(a1,2)));
%     [ec, counter] = palette_wl(a1, x1);
%     Counter = [Counter, counter];
%     counter
%     names1 = num2str(ec);
%     names = strcat(names1, ' (', names0, ')');
%     names = cellstr(names);
%     g1 = graph(a1);
%     figure(i);
%     plot(g1, 'NodeLabel', names);
%     title(Y(j));
%     input('next?');
end

% maximum WL convergence iteration number
% max(Counter)
% mean(Counter)
max(nodeNum)
maxNodeDeg
