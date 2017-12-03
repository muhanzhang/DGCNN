-- Usage: a gradient checking script testing the backpropagation function of GraphConv layer.

require 'nn'
require 'GraphConv'
require 'GraphReLU'

--[[
model = nn.Sequential()
model:add(nn.GraphConv(2, 2, 1, true))
--model:add(nn.GraphReLU())

A = torch.Tensor({{1, 0}, {0, 1}})
x = torch.Tensor({{1, -1}, {-1, 1}}):reshape(2, 2, 1)
print(A)
print(x)
par, gpar = model:getParameters()
print(par)
print(gpar)
b = model:forward({A, x})
print(b[1])
print(b[2])



dx = torch.Tensor({{1, -1}, {0, 1}}):reshape(2, 2, 1)
c = model:backward({A, x}, dx)
print(dx)
--print(par)
--print(gpar)
print(c)
]]




-- TEST GRADIENT --

-- parameters
local precision = 1e-5
require 'Jacobian1'
local jac = nn.Jacobian

-- define inputs and module
local a = 10
local b = 20
local O = math.random(a, b)
local I = math.random(a, b)
local n = math.random(a, b)
local N = math.random(a, b)

-- test single instance
local input = {}
input[1] = torch.randn(n, n)
input[2] = torch.Tensor(n,I):zero()
local module = nn.GraphConv(I, O)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end


-- test batch mode
local input = {}
input[1] = torch.randn(N, n, n)
input[2] = torch.Tensor(N, n, I):zero()
local module = nn.GraphConv(I, O)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
