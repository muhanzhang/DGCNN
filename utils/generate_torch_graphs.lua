-- Usage: for converting .mat graphs in data/raw_data into .dat format read by torch
-- require 'matio' installed
-- *author: Muhan Zhang, Washington University in St. Louis

matio = require 'matio'
require 'paths'

Datasets = {'DD', 'MUTAG', 'NCI1', 'ptc', 'proteins', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI'}

for _, dataname in pairs(Datasets) do
   print(dataname)
   local instance = {}
   local label = {}
   local datapath = '../data/raw_data/'
   local tmp = matio.load(datapath..dataname..'.mat')
   local tmp_label = tmp[string.lower('l'..dataname)]
   local tmp_instance = tmp[dataname]

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
      if #ins['nl'] == 0 then -- empty
         if #ins['na'] ~= 0 then
            local node_information = ins['na'].values
         else
            local node_information = {}
         end
      else
         local node_information = ins['nl'].values
      end

      local tmp1 = {ins['am'], node_information}
      local tmp2 = tmp_label[i][1]
      instance[i] = tmp1
      label[i] = tmp2
   end
   local dataset = {instance = instance, label = label}
   torch.save('../data/'..dataname..'.dat', dataset)

end
