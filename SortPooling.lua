-- SortPooling layer implementation --
-- -k: pooling parameter, number of vertices to keep
-- -sortChannel: sort vertices according to this feature channel
-- Note that for efficiency reason, we do not break ties by further sorting other channels.
--
-- *author: Muhan Zhang, Washington University in St. Louis


local SortPooling, parent = torch.class('nn.SortPooling', 'nn.Module')

function SortPooling:__init(k, sortChannel)
   parent.__init(self)
   self.k = k
   self.sc = sortChannel or -1  -- sort all n vertices according to this Channel, default the last Channel
end

function SortPooling:updateOutput(input)
   assert(input:dim() <= 3, 'Only 2D [n * channel] or 3D [N * n * channel] accepted')
   local k = self.k
   local sc = self.sc
   if input:dim() == 2 then
      local nodeDim = 1 -- node dimension
      local n = input:size(1)
      local channel = input:size(2)
      -- sort the last channel of input in descending order
      local sorted, allIndices = input[{{}, {sc}}]:sort(1, true)
      allIndices = torch.repeatTensor(allIndices, 1, channel)
      -- reduce the indices to only include the top-k
      if k > n then k = n end
      self.indices = allIndices:narrow(nodeDim, 1, k)
      self.output = input:gather(nodeDim, self.indices)
      if self.k > k then
         self.output = torch.cat(self.output, torch.zeros(self.k - k, channel):typeAs(self.output), nodeDim)
      end
   elseif input:dim() == 3 then  -- batch mode
      local nodeDim = 2 -- node dimension
      local N, n = input:size(1), input:size(2)
      local channel = input:size(3)
      -- sort the last channel of input in descending order
      local sorted, allIndices = input[{{}, {}, {sc}}]:sort(2, true)
      allIndices = torch.repeatTensor(allIndices, 1, 1, channel)
      -- reduce the indices to only include the top-k
      if k > n then k = n end
      self.indices = allIndices:narrow(nodeDim, 1, k)
      self.output = input:gather(nodeDim, self.indices)
      if self.k > k then
         self.output = torch.cat(self.output, torch.zeros(N, self.k - k, channel):typeAs(self.output), nodeDim)
      end
   end
   return self.output
end

function SortPooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      if input:dim() == 2 then
         local nodeDim = 1
         local n = input:size(1)
         local channel = input:size(2)
         -- scatter gradOutput to original indices
         local k = math.min(self.k, n)
         local updateValues = self.gradInput:gather(nodeDim, self.indices)
         updateValues:add(gradOutput:narrow(nodeDim, 1, k))
         self.gradInput:scatter(nodeDim, self.indices, updateValues)
      elseif input:dim() == 3 then  -- batch mode
         local nodeDim = 2
         local N, n = input:size(1), input:size(2)
         local channel = input:size(3)
         -- scatter gradOutput to original indices
         local k = math.min(self.k, n)
         local updateValues = self.gradInput:gather(nodeDim, self.indices)
         updateValues:add(gradOutput:narrow(nodeDim, 1, k))
         self.gradInput:scatter(nodeDim, self.indices, updateValues)
      end
      return self.gradInput
   end
end

function SortPooling:clearState()
   nn.utils.clear(self, 'indices')
   return parent.clearState(self)
end
