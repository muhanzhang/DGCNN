local GraphReLU, Parent = torch.class('nn.GraphReLU', 'nn.Threshold')

function GraphReLU:__init(p)
   Parent.__init(self,0,0,p)
   self.output = {}
end

function GraphReLU:updateOutput(input)
   self:validateParameters()
   self.output[1] = input[1]
   self.output[2] = torch.Tensor():typeAs(input[2])
   input[2].THNN.Threshold_updateOutput(
      input[2]:cdata(),
      self.output[2]:cdata(),
      self.threshold,
      self.val,
      self.inplace
   )
   return self.output
end

function GraphReLU:updateGradInput(input, gradOutput)
   self:validateParameters()
   input[2].THNN.Threshold_updateGradInput(
      input[2]:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.threshold,
      self.val,
      self.inplace
   )
   return self.gradInput
end
