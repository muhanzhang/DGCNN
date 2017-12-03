local GraphSoftMax, _ = torch.class('nn.GraphSoftMax', 'nn.Module')

function GraphSoftMax:updateOutput(input)
   self.output = {}
   self.output[1] = input[1]
   self.output[2] = torch.Tensor():typeAs(input[2])
   input[2].THNN.SoftMax_updateOutput(
      input[2]:cdata(),
      self.output[2]:cdata()
   )
   return self.output
end

function GraphSoftMax:updateGradInput(input, gradOutput)
   input[2].THNN.SoftMax_updateGradInput(
      input[2]:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output[2]:cdata()
   )
   return self.gradInput
end
