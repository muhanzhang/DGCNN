-- Usage: a gradient checking script testing the backpropagation function of SortPooling.

require 'nn'
require 'SortPooling'

--[[
model = nn.Sequential()
model:add(nn.SortPooling(4))

a = torch.Tensor():randn(3, 3, 3)
print(a)
b = model:forward(a)
print(b)
c = torch.Tensor(3, 3, 4):fill(1)
c[1][1] = torch.Tensor({1, -1, 1, -1})
c[2][1] = torch.Tensor({-1, 1, -1, 1})
d = model:backward(a, c)
print(d)
]]


-- TEST GRADIENT --

-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local a = 10
local b = 20
local N = math.random(a, b)
local n = math.random(a, b)
local I = math.random(a, b)

local input = torch.Tensor(n, I):zero()
local module = nn.SortPooling(15)  -- change to 25 to test behavior when k < d
-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end

-- test batch mode
local input = torch.Tensor(N, n, I):zero()
local module = nn.SortPooling(15)  -- change to 25 to test behavior when k < d

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end
