-- Usage: for converting .mat graphs in data/raw_data into .dat format read by torch
-- require 'matio' installed
-- *author: Muhan Zhang, Washington University in St. Louis

matio = require 'matio'
require 'paths'

Datasets = {'MUTAG', 'DD', 'NCI1', 'ptc', 'proteins', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI'}

for _, dataname in pairs(Datasets) do
   print(dataname)
   local instance = {}
   local label = {}
   local datapath = '../data/raw_data/'
   local tmp = matio.load(datapath..dataname..'.mat')
   local tmp_label = tmp[string.lower('l'..dataname)]:type('torch.ByteTensor') -- convert to bytetensor to save space
   local tmp_instance = tmp[dataname][1]

   -- transform labels to standard 1, 2, ..., n classes
   local label_map = {}
   for i = 1, tmp_label:size(1) do
      label_map[tmp_label[i][1]] = tmp_label[i][1]
   end
   local unique_label = {}
   for k, v in pairs(label_map) do
      table.insert(unique_label, k)
   end
   table.sort(unique_label)
   local label_map = {}
   for k, v in pairs(unique_label) do
      label_map[v] = k
   end
   for i = 1, tmp_label:size(1) do
      local l = tmp_label[i][1]
      tmp_label[i][1] = label_map[l]
   end

   -- save in torch serialization
   for i = 1, tmp_label:size(1) do
      if i % 100 == 0 then print(i) end
      local ins = tmp_instance[i]
      local node_information = {}
      if next(ins.nl)~=nil then -- non-empty
         node_information = ins['nl'].values:type('torch.ByteTensor')
      else
         node_information = torch.ones(ins.am:size(1), 1):type('torch.ByteTensor')
      end

      local tmp1 = {}
      tmp1[1] = ins['am']:type('torch.ByteTensor')
      tmp1[2] = node_information
      local tmp2 = tmp_label[i][1]
      instance[i] = tmp1
      label[i] = tmp2
   end
   local dataset = {instance = instance, label = label}
   torch.save('../data/'..dataname..'.dat', dataset)

end
