local EdgeDropout, Parent = torch.class('nn.EdgeDropout', 'nn.Module')
function EdgeDropout:__init(p,v1,inplace,stochasticInference)
   Parent.__init(self)
self.p = p or 0.5
self.train = true
self.inplace = inplace
self.stochastic_inference = stochasticInference or false
-- version 2 scales output during training instead of evaluation
self.v2 = not v1
self.output = {}
if self.p >= 1 or self.p < 0 then
error('<EdgeDropout> illegal percentage, must be 0 <= p < 1')
end
self.noise = torch.Tensor()
end
function EdgeDropout:updateOutput(input)
local A = input[1]
local x = input[2]
if self.inplace then
self.output:set(input)
else
self.output[1] = torch.Tensor():typeAs(A):resizeAs(A):copy(A)
self.output[2] = x
end
if self.p > 0 then
if self.train or self.stochastic_inference then
self.noise:resizeAs(A)
self.noise:bernoulli(1-self.p)
if self.v2 then
self.noise:div(1-self.p)
end
self.output[1]:cmul(self.noise)
elseif not self.v2 then
self.output[1]:mul(1-self.p)
end
end
return self.output
end
function EdgeDropout:updateGradInput(input, gradOutput)
self.gradInput = gradOutput
return self.gradInput
end
function EdgeDropout:setp(p)
self.p = p
end
function EdgeDropout:__tostring__()
return string.format('%s(%f)', torch.type(self), self.p)
end
function EdgeDropout:clearState()
if self.noise then
self.noise:set()
end
return Parent.clearState(self)
end
