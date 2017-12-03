local GraphTanh = torch.class('nn.GraphTanh', 'nn.Module')

function GraphTanh:updateOutput(input)
   self.output = {}
   self.output[1] = input[1]
   self.output[2] = torch.Tensor():typeAs(input[2])
   input[2].THNN.Tanh_updateOutput(
      input[2]:cdata(),
      self.output[2]:cdata()
   )
   return self.output
end

function GraphTanh:updateGradInput(input, gradOutput)
   input[2].THNN.Tanh_updateGradInput(
      input[2]:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output[2]:cdata()
   )
   return self.gradInput
end
