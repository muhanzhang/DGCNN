-- Usage: a gradient checking script testing the backpropagation function of graph transfer function layers.

require 'nn'
require 'GraphReLU'
require 'GraphSoftMax'
local precision = 1e-5
require 'Jacobian1'
local jac = nn.Jacobian

-- test GraphReLU

-- define inputs and module
local a = 10
local b = 20
local O = math.random(a, b)
local I = math.random(a, b)
local n = math.random(a, b)
local d = math.random(a, b)
local input = {}
input[1] = torch.randn(n, n)
input[2] = torch.Tensor(I,n,d):zero()
local module = nn.GraphReLU()

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end

-- test GraphSoftmax

-- define inputs and module
local a = 10
local b = 20
local O = math.random(a, b)
local I = math.random(a, b)
local n = math.random(a, b)
local d = math.random(a, b)
local input = {}
input[1] = torch.randn(n, n)
input[2] = torch.Tensor(I,n,d):zero()
local module = nn.GraphSoftMax()

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end

