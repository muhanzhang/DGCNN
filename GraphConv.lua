-- Graph convolution with multiple input and output channels --
-- input[1]=output[1]=A: (N)*n*n, N if batch. 
-- input[2]=x: (N)*n*I; output[2]=x': (N)*n*O
-- -N: batch size, -n: node number, -I: input channel, -O: output channel
--
-- *author: Muhan Zhang, Washington University in St. Louis

local GraphConv, parent = torch.class('nn.GraphConv', 'nn.Module')

function GraphConv:__init(I, O, bias, fixedWeight)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   local fixedWeight = fixedWeight or false
   self.weight = torch.Tensor(I, O)
   self.gradWeight = torch.Tensor(I, O)
   if bias then
      self.bias = torch.Tensor(O)
      self.gradBias = torch.Tensor(O)
   end
   if fixedWeight then
      self.fixedWeight = fixedWeight
   end
      
   self.I = I
   self.O = O
   self.output = {}
   self:reset()
end

function GraphConv:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function GraphConv:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.I / 2)
   end
   self.weight:uniform(-stdv, stdv)
   if self.bias then 
      self.bias:uniform(-stdv, stdv) 
   end
   if self.fixedWeight then
      self.weight:fill(self.fixedWeight)
      if self.bias then self.bias:fill(0) end
   end
   return self
end

function GraphConv:updateOutput(input)
   local A = input[1]
   local x = input[2]
   local w = self.weight
   self.output[1] = A
   self.output[2] = torch.Tensor():typeAs(x)
   if x:dim() == 2 then  -- single instance
      local n = x:size(1)
      self.output[2]:resize(n, self.O):zero()
      local res = (x * w)
      if self.bias then res:add(self.bias:repeatTensor(n, 1)) end
      res = A * res
      self.output[2]:add(res)
      return self.output
   elseif x:dim() == 3 then  -- batch mode
      local N, n = x:size(1), x:size(2)
      self.output[2]:resize(N, n, self.O):zero()
      local res = (x:reshape(N*n, self.I) * w):reshape(N, n, self.O)
      if self.bias then res:add(self.bias:repeatTensor(N, n, 1)) end
      res2 = torch.bmm(A, res)
      self.output[2]:add(res2)
      return self.output
   end
end

function GraphConv:updateGradInput(input, gradOutput)
   local A = input[1]
   local x = input[2]
   local w = self.weight
   if self.gradInput then
      self.gradInput:resizeAs(x):zero()
      if x:dim() == 2 then
         local n = x:size(1)
         local dx = A:t() * gradOutput * w:t()
         self.gradInput:add(dx)
         return self.gradInput
      elseif x:dim() == 3 then  -- batch mode
         local N, n = x:size(1), x:size(2)
         local tmp = (gradOutput:reshape(N*n, self.O) * w:t()):reshape(N, n, self.I)
         local dx = torch.bmm(A:transpose(2, 3), tmp)
         self.gradInput:add(dx)
         return self.gradInput
      end
   end
end

function GraphConv:accGradParameters(input, gradOutput, scale)
   if not self.fixedWeight then
      local A = input[1]
      local x = input[2]
      scale = scale or 1
      if x:dim() == 2 then
         local n = x:size(1)
         local res = x:t() * A:t() * gradOutput
         self.gradWeight:add(scale * res)
         if self.bias then
            local res2 = A:t() * gradOutput
            res2 = (torch.ones(1, n):typeAs(res2) * res2):resizeAs(self.bias)
            self.gradBias:add(scale * res2)
         end
      elseif x:dim() == 3 then
         local N, n = x:size(1), x:size(2)
         local res = torch.bmm(x:transpose(2, 3), A:transpose(2, 3))
         local res2 = torch.zeros(self.I, self.O):typeAs(res)
         res2:addbmm(res, gradOutput)
         self.gradWeight:add(scale * res2)
         if self.bias then
            local res2 = torch.zeros(n, self.O):typeAs(A)
            res2:addbmm(A:transpose(2, 3), gradOutput)
            res2 = (torch.ones(1, n):typeAs(res2) * res2):resizeAs(self.bias)
            self.gradBias:add(scale * res2)
         end
      end
   end
end

GraphConv.sharedAccUpdateGradParameters = GraphConv.accUpdateGradParameters

function GraphConv:clearState()
   return parent.clearState(self)
end

function GraphConv:__tostring__()
   return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end

