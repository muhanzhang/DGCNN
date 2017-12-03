-- Main program of DGCNN.
-- *author: Muhan Zhang, Washington University in St. Louis

require 'paths'
require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'optim'
require 'SortPooling'
require 'GraphConv'
require 'GraphReLU'
require 'GraphTanh'
require 'EdgeDropout'
require 'GraphSelectTable'
require 'GraphConcatTable'

------------------------------------------------------------------------
--                              Parser                                --
------------------------------------------------------------------------

local function commandLine()
   local cmd = torch.CmdLine()

   cmd:text()
   cmd:text('Options:')
   -- general options
   cmd:option('-seed', 		      100,           'fixed input seed for repeatable experiments')
   cmd:option('-debug', 	      false,         'debug mode (output intermediate results after each training epoch)')
   cmd:option('-fixed_shuffle',   'none',        'x_y means using data/shuffle/$dataNamex_y.mat as fixed shuffle indices; otherwise use random shuffle indices')
   cmd:option('-ensemble',        0,             'if x~=0, use the intermediate nets every x epochs as an ensemble. Using ensemble needs to set -valRatio 0')
   -- dataset options
   cmd:option('-dataName',        'MUTAG',       'Specify which dataset to use')
   cmd:option('-nClass',          2,             'Specify # of classes of dataset')
   cmd:option('-trainRatio', 	  .8,            'Specify size of train set')
   cmd:option('-valRatio', 	      .1,            'Specify size of validation set')
   cmd:option('-maxNodeLabel',    7,             'Specify maximum node label, required if nodeLabel = oneHot')
   -- graph convolution settings
   cmd:option('-bias',            false,         'Whether to include bias b in A(XW+b)')
   cmd:option('-convMatrix',      'rwAplusI',    'Specify which propagation model to use: symAplusI, AplusI, A, rwAplusI')
   cmd:option('-alpha',           1,             'Specify the relative weight of A to I, i.e., I + alpha * A')
   cmd:option('-nodeLabel',       'oneHot',      'Specify node label encoding schemes: original, allOne, nDegree, oneHot, oneHot+nDegree')
   cmd:option('-shareConv',       false,         'whether to share parameters of GraphConv layers')
   cmd:option('-originalFeature', false,         'whether to add original node features into GraphConv feature vectors')
   cmd:option('-inputChannel',    0,             'Specify # of input channels of the first GraphConv layer. If nodeLabel = original, then this must be specified; otherwise this will be automatically set.')
   cmd:option('-outputChannels',  '32 32 32 1',  'Specify # of output channels of GraphConv layers')
   cmd:option('-nonlinear',       'tanh',        'Specify which nonlinearity to use between GraphConv: relu, tanh, softmax, no')
   cmd:option('-oneWeight',       false,         'whether to use a fixed weight 1 in GraphConv layers')
   cmd:option('-edgeDropout',     0,             'randomly drop out some edges after each GraphConv')
   -- SortPooling options
   cmd:option('-noSortPooling',   false,         'no SortPooling, performs only pooling without sorting.')
   cmd:option('-sumNodeFeatures', false,         'no SortPooling, direclty sum node features followed by only dense layers.')
   cmd:option('-k',               0,             'Specify k in SortPooling layer. If not specified, k is set so that 60% graphs in the dataset have size >= k.')
   -- 1-D convolution and fully-connected layers' settings
   cmd:option('-TCChannels',      '16 32',       'Specify the # of channels of the 1-D temporal convolution layers')
   cmd:option('-TCkw',            '0 5',         'Specify the kernel width of temporal convolution layers, 0 means to use the total # of outputChannels of the effective GraphConv layers (only for first layer)')
   cmd:option('-TCdropout',       false,         'whether to use dropout in temporal convolution layers')
   cmd:option('-MPkw',            2,             'Specify the kernel width of max pooling layers')
   cmd:option('-firstTCnoMP',     false,         'if true, will not add max pooling after the first TC layer')
   cmd:option('-nhLayers',        1,             'number of fully-connected layers after TC layers')
   cmd:option('-nhu', 		      128,           'number of hidden units in fully-connected layers')
   cmd:option('-dropout', 		  0.5,           'Specify dropout rate of fully-connected layers')
   -- optimization options
   cmd:option('-batch',           false,         'whether to use mini-batch gradient descent')
   cmd:option('-batchSize',       16,            'mini-batch size')
   cmd:option('-no_shuffle',      false,         'whether to not shuffle training data in each epoch')
   cmd:option('-optimization',    'ADAM',        'Specify optimization method: ADAM, SGD')
   cmd:option('-learningRate',    0.0001,        'learning rate at t=0')
   cmd:option('-halfLR', 	      false,         'whether to half learning rate when training stucks')
   cmd:option('-earlyStop',       false,         'whether early stop training when stucks')
   cmd:option('-decay_lr', 	      1e-6,          'learning rate decay (SGD only)')
   cmd:option('-momentum', 	      0.9,           'momentum (SGD only)')
   cmd:option('-l2reg', 		  0,             'l2 regularization is not recommended since it will update weights of GraphConv layers > r too')
   cmd:option('-maxEpoch', 	      300,           'maximum # of epochs to train for')
   cmd:option('-save', 	          'result',      'result saving position')
   cmd:option('-gpu', 	          1,             'Specify default GPU')
   cmd:option('-log', 	          false,         'whether to log all screen outputs')
   cmd:option('-repeatSave',      true,          'whether to append final results of each run to a file every time for repeated experiments')

   cmd:text()

   local opt = cmd:parse(arg or {})

   if opt.fixed_shuffle ~= 'none' then
      torch.manualSeed(opt.seed) -- fixed seed and fixed shuffle for repeatable experiments
      cutorch.manualSeedAll(opt.seed) 
      matio = require 'matio'
      tmp = matio.load('data/shuffle/'..opt.dataName..opt.fixed_shuffle..'.mat')
      opt.shuffle_idx = tmp.r_current
   end

   if opt.debug then 
      opt.trainRatio = 1
      opt.valRatio = 0
   end
   
   if opt.log then -- log position = result/dataName/log
      opt.logName = cmd:string('exp', opt, {dir=true})
      local logDir = paths.concat(opt.save, opt.dataName, 'log', opt.logName)
      os.execute('mkdir -p ' .. paths.dirname(logDir))
      cmd:log(logDir, opt)
   end
   
   if opt.nonlinear == 'relu' then
      opt.nonlinear = nn.GraphReLU
   elseif opt.nonlinear == 'tanh' then
      opt.nonlinear = nn.GraphTanh
   elseif opt.nonlinear == 'softmax' then
      opt.nonlinear = nn.GraphSoftMax
   elseif opt.nonlinear == 'no' then
      opt.nonlinear = nn.Identity
   end

   if opt.convMatrix == 'symAplusI' then
      alpha = opt.alpha
      convMatrix = symAplusI
   elseif opt.convMatrix == 'AplusI' then
      alpha = opt.alpha
      convMatrix = AplusI
   elseif opt.convMatrix == 'A' then
      convMatrix = function(A) return A end
   elseif opt.convMatrix == 'rwAplusI' then
      alpha = opt.alpha
      convMatrix = rwAplusI
   end

   if opt.nodeLabel == 'original' then
      processNodeLabel = function(x) return x end
      assert(opt.inputChannel > 0, 'must specify number of input channels when using original node features')
   elseif opt.nodeLabel == 'allOne' then
      processNodeLabel = allOne
      opt.inputChannel = 1
   elseif opt.nodeLabel == 'oneHot' then
      maxNodeLabel = opt.maxNodeLabel
      opt.inputChannel = maxNodeLabel  -- change inputChannel to # of one-hot bits
      processNodeLabel = oneHot
   elseif opt.nodeLabel == 'nDegree' then
      processNodeLabel = normalizedDegree
      opt.inputChannel = 1
   elseif opt.nodeLabel == 'oneHot+nDegree' then
      processNodeLabel = oneHotnDegree
      maxNodeLabel = opt.maxNodeLabel
      opt.inputChannel = maxNodeLabel + 1
   end

   local tmp = {}
   opt.totalOutputChannels = 0
   if opt.originalFeature then opt.totalOutputChannels = opt.inputChannel end
   local layerCount = 0
   for i in string.gmatch(opt.outputChannels, "%S+") do
      layerCount = layerCount + 1
      table.insert(tmp, tonumber(i))
      opt.totalOutputChannels = opt.totalOutputChannels + tonumber(i)
   end
   opt.outputChannels = tmp
   opt.nGLayers = #opt.outputChannels  -- # of GraphConv layers
   opt.effectiveTotalOutputChannels = opt.totalOutputChannels

   local tmp = {}
   for i in string.gmatch(opt.TCChannels, "%S+") do
      table.insert(tmp, tonumber(i))
   end
   opt.TCChannels = tmp
   
   local tmp = {}
   for i in string.gmatch(opt.TCkw, "%S+") do
      table.insert(tmp, tonumber(i))
   end
   opt.TCkw = tmp
   
   if opt.TCkw[1] == 0 then
      opt.TCkw[1] = opt.effectiveTotalOutputChannels
   end

   if opt.batch == false then
      opt.batchSize = 1
   else
      opt.learningRate = opt.learningRate * opt.batchSize
   end

   if opt.optimization == 'SGD' then
      opt.optimize = optim.sgd
   elseif opt.optimization == 'ADAM' then
      opt.optimize = optim.adam
   elseif opt.optimization == 'RMSPROP' then
      opt.optimize = optim.rmsprop
   else
      error('unknown optimization method')
   end


   return opt
end
------------------------------------------------------------------------
--                               Model                                --
------------------------------------------------------------------------

local function create_model(opt)
   local opc = opt.outputChannels
   opc[0] = opt.inputChannel
   net = nn.Sequential()

   -- Recurrent GraphConv layers
   local c0 = nn.Sequential()  -- the whole graph convolution structure
   local c = {}  -- one condition of a branch
   local b = {}  -- branches
   for i = 1, opt.nGLayers do
      -- add recurrent units from last to first
      local j = opt.nGLayers - i
      c[j] = nn.Sequential()
      if opt.shareConv then -- if opt.sharConv, must use exactly the same graph conv layers
                            -- i.e., the same # of input and output channels in each layer
         if i == 1 then -- the last recurrent unit
            c[j]:add(nn.GraphConv(opc[j], opc[j+1]), opt.bias)
            c[j]:add(opt.nonlinear())
            c[j]:add(nn.GraphSelectTable(2))
         else
            local convCopy = c[j+1].modules[1]:clone('weight', 'bias', 'gradWeight', 'gradBias')
            c[j]:add(convCopy)
            c[j]:add(opt.nonlinear())
            c[j]:add(b[j+1])
         end

      else
         if i == 1 then  -- last recurrent unit does not have any nonlinearity, or have?
            if opt.oneWeight then
               c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias, 1))
            else
               c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias))
            end
            c[j]:add(opt.nonlinear())  -- add to see performance
            c[j]:add(nn.GraphSelectTable(2))
         elseif j > opt.nGLayers - 1 then
            if opt.oneWeight then  -- not use random weight, but fixed weight = 1, bias = 0 in layers > H, and use tanh
               c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias, 1))
               --c[j]:add(nn.GraphTanh())
               c[j]:add(opt.nonlinear())
            else
               c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias))
               c[j]:add(opt.nonlinear())
            end
            c[j]:add(b[j+1])
         else             -- only first few effective graph convolution layers (j=0, 1, ..., r-1) have user-defined nonlinearity and EdgeDropout
            if opt.oneWeight then
               c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias, 1))
            else
               c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias))
            end
            if opt.edgeDropout ~= 0 then
               c[j]:add(nn.EdgeDropout(opt.edgeDropout))
            end
            c[j]:add(opt.nonlinear())
            c[j]:add(b[j+1])
         end
      end
      
      if j == 0 then
         if opt.originalFeature then
            b0 = nn.GraphConcatTable()
            local tmp = nn.Sequential()
            tmp:add(nn.GraphSelectTable(2))
            b0:add(tmp)
            b0:add(c[j])
            c0:add(b0)
         else
            c0 = c[j]
         end
         break
      end
      b[j] = nn.GraphConcatTable()
      local tmp = nn.Sequential()
      tmp:add(nn.GraphSelectTable(2))
      b[j]:add(tmp)
      b[j]:add(c[j])
   end
   

   -- combine outputs of all recurrent units
   net:add(c0)
   net:add(nn.FlattenTable())
   net:add(nn.JoinTable(3))
   if opt.noSortPooling then
      net:add(nn.Padding(2, opt.k))
      net:add(nn.Narrow(2, 1, opt.k))
   elseif opt.sumNodeFeatures then
      net:add(nn.Sum(1, 2)) -- sum all node features as a graph-level feature
   else
      net:add(nn.SortPooling(opt.k))
   end

   if opt.sumNodeFeatures == false then
      -- now input becomes a (K * totalOutputChannels) tensor
      net:add(nn.View(-1, opt.k * opt.totalOutputChannels, 1))
      -- 1-D convolution layers
      net:add(nn.TemporalConvolution(1, opt.TCChannels[1], opt.TCkw[1], opt.totalOutputChannels))  -- now K * TCChannels[1]
      net:add(nn.ReLU())
      local nFrame = opt.k  -- record number of frames (vertices) after each conv
      for i = 1, #(opt.TCChannels)-1 do
         if opt.firstTCnoMP then
            if i > 1 then
               net:add(nn.TemporalMaxPooling(opt.MPkw, opt.MPkw))
               nFrame = (nFrame - opt.MPkw) / opt.MPkw + 1
               nFrame = math.floor(nFrame)
            end
         else
            net:add(nn.TemporalMaxPooling(opt.MPkw, opt.MPkw))
            nFrame = (nFrame - opt.MPkw) / opt.MPkw + 1
            nFrame = math.floor(nFrame)
         end 
         net:add(nn.TemporalConvolution(opt.TCChannels[i], opt.TCChannels[i+1], opt.TCkw[i+1], 1))
         net:add(nn.ReLU())
         if opt.TCdropout then
            net:add(nn.Dropout(0.5))
         end
         nFrame = (nFrame - opt.TCkw[i+1]) / 1 + 1
      end
      -- now nFrame * TCChannels[-1]
      prev = nFrame * opt.TCChannels[#(opt.TCChannels)]
      net:add(nn.View(-1, prev))
   else -- use summed node features directly
      prev = opt.totalOutputChannels
   end

   -- fully connected layers
   for i = 1, opt.nhLayers do
      net:add(nn.Linear(prev, opt.nhu))
      net:add(nn.ReLU())
      net:add(nn.Dropout(opt.dropout))
      prev = opt.nhu
   end
   net:add(nn.Linear(prev, opt.nClass))
   net:add(nn.LogSoftMax())
   net = net:cuda()

   -- Criterion
   criterion = nn.ClassNLLCriterion()
   criterion = criterion:cuda()
   print(net)
end
------------------------------------------------------------------------
--                           Data Loader                              --
------------------------------------------------------------------------

local function load_data(opt)
   dataname = opt.dataName
   local dataset = torch.load('data/'..dataname..'.dat')
   local train_ratio = opt.trainRatio
   local validation_ratio = opt.valRatio
   local N = #dataset.label
   local Ntrain = math.ceil(N * train_ratio)
   local Nvalidation = math.ceil(N * (train_ratio + validation_ratio)) - Ntrain
   local Ntest = N - Ntrain - Nvalidation
   local shuffle_idx = torch.randperm(N)
   if opt.fixed_shuffle ~= 'none'  then
      shuffle_idx = opt.shuffle_idx:resizeAs(shuffle_idx)
   end
   if opt.debug then   
      shuffle_idx = torch.Tensor(#dataset.label)
      for j = 1, #dataset.label do shuffle_idx[j] = j end
   end

   -- randomly split into train, test, val
   trainset = {instance = {}, label = {}, ns = {}}
   valset = {instance = {}, label = {}, ns = {}}
   testset = {instance = {}, label = {}, ns = {}}
   local Ns = torch.zeros(N) -- record the size of each graph
   for i = 1, Ntrain do
      trainset.instance[i] = dataset.instance[shuffle_idx[i]]
      trainset.instance[i][2] = processNodeLabel(trainset.instance[i][2], trainset.instance[i][1])
      trainset.instance[i][1] = convMatrix(trainset.instance[i][1])
      trainset.label[i] = dataset.label[shuffle_idx[i]]
      local tmp = trainset.instance[i][1]:size(1)  -- ns: for recording the sizes of graphs
      trainset.ns[i] = tmp
      Ns[i] = tmp
   end
   for i = Ntrain+1, Ntrain+Nvalidation do
      valset.instance[i - Ntrain] = dataset.instance[shuffle_idx[i]]
      valset.instance[i - Ntrain][2] = processNodeLabel(valset.instance[i - Ntrain][2], valset.instance[i - Ntrain][1])
      valset.instance[i - Ntrain][1] = convMatrix(valset.instance[i - Ntrain][1])
      valset.label[i - Ntrain] = dataset.label[shuffle_idx[i]]
      local tmp = valset.instance[i - Ntrain][1]:size(1)
      valset.ns[i - Ntrain] = tmp
      Ns[i] = tmp
   end
   for i = Ntrain+Nvalidation+1, N do
      testset.instance[i - Ntrain - Nvalidation] = dataset.instance[shuffle_idx[i]]
      testset.instance[i - Ntrain - Nvalidation][2] = processNodeLabel(testset.instance[i - Ntrain - Nvalidation][2], testset.instance[i - Ntrain - Nvalidation][1])
      testset.instance[i - Ntrain - Nvalidation][1] = convMatrix(testset.instance[i - Ntrain - Nvalidation][1])
      testset.label[i - Ntrain - Nvalidation] = dataset.label[shuffle_idx[i]]
      local tmp = testset.instance[i - Ntrain - Nvalidation][1]:size(1)
      testset.ns[i - Ntrain - Nvalidation] = tmp
      Ns[i] = tmp
   end
   if opt.k == 0 then
      local tmp = torch.sort(Ns)
      opt.k = tmp[math.ceil(0.6 * N)]
      print(opt.k)
   end

end

------------------------------------------------------------------------
--                         Utility Functions                          --
------------------------------------------------------------------------

-- propagation model
function symAplusI(A)
   local n = A:size(1)
   local A_tilde = alpha * A + torch.eye(n, n)
   local D_tilde = A_tilde:sum(2):resize(n)
   D_tilde:pow(-0.5)
   D_tilde = torch.diag(D_tilde)
   local L = D_tilde * A_tilde * D_tilde
   return L
end

function rwAplusI(A)
   local n = A:size(1)
   local A_tilde = alpha * A + torch.eye(n, n)
   local D_tilde = A_tilde:sum(2):resize(n)
   D_tilde:pow(-1)
   D_tilde = torch.diag(D_tilde)
   local L = D_tilde * A_tilde
   return L
end

function AplusI(A)
   local n = A:size(1)
   local A_tilde = alpha * A + torch.eye(n, n)
   return A_tilde
end

-- process node labels
function allOne(x)
   return torch.Tensor():resizeAs(x):fill(1)
end

function oneHot(x)
   -- need to specify maxNodeLabel initially
   local indices = x:type('torch.LongTensor')
   local one_hot = torch.zeros(indices:size(1), maxNodeLabel)
   one_hot:scatter(2, indices, 1)
   return one_hot
end

function normalizedDegree(x, A)
   local Degree = A:sum(2):resizeAs(x)
   local nDegree = Degree:div(x:size(1) - 1)
   return nDegree
end

function oneHotnDegree(x, A)
   -- need to specify maxNodeLabel initially
   local indices = x:type('torch.LongTensor')
   local one_hot = torch.zeros(indices:size(1), maxNodeLabel + 1)
   one_hot:scatter(2, indices, 1)
   local Degree = A:sum(2):resizeAs(x)
   local nDegree = Degree:div(x:size(1) - 1)
   one_hot[{{}, {-1}}] = nDegree
   return one_hot
end
------------------------------------------------------------------------
--                              Training                              --
------------------------------------------------------------------------

-- training function
function train(dataset)
   net:training()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch)
   if opt.no_shuffle then
      shuffle = torch.Tensor(#dataset.label)
      for j = 1, #dataset.label do shuffle[j] = j end
   elseif opt.debug then  -- arrange the first, second, and last sample to top and show them in debug
      shuffle = torch.Tensor(#dataset.label)
      for j = 4, #dataset.label do shuffle[j] = j-1 end
      shuffle[1] = 1
      shuffle[2] = 2
      shuffle[3] = #dataset.label
   else  -- not in debug mode, and random shuffle is turned on
      shuffle = torch.randperm(#dataset.label)
   end
   local grad_over_param = 0
   for t = 1, #dataset.label, opt.batchSize do
      -- disp progress
      xlua.progress(t, #dataset.label)
      -- bound batchSize
      local batchSize
      if t + opt.batchSize - 1 > #dataset.label then
         batchSize = #dataset.label - t + 1
      else
         batchSize = opt.batchSize
      end
      local nMax = 0
      -- record the maximum graph size in the mini-batch
      for i = t, t + batchSize - 1 do
         if dataset.ns[shuffle[i]] > nMax then nMax = dataset.ns[shuffle[i]] end
      end
      -- create the mini-batch
      local inputs = {}
      inputs[1] = torch.zeros(batchSize, nMax, nMax):cuda()
      inputs[2] = torch.zeros(batchSize, nMax, dataset.instance[1][2]:size(2)):cuda()
      local targets = torch.Tensor(batchSize):cuda()
      local batchCount = 0
      for i = t, t + batchSize - 1 do
         batchCount = batchCount + 1
         local A = dataset.instance[shuffle[i]][1]
         local x = dataset.instance[shuffle[i]][2]
         local y = dataset.label[shuffle[i]]
         inputs[1][{{batchCount}, {1, A:size(1)}, {1, A:size(2)}}] = A
         inputs[2][{{batchCount}, {1, x:size(1)}, {1, x:size(2)}}] = x
         targets[batchCount] = y
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         local output = net:forward(inputs)
         local f = criterion:forward(output, targets)
         trainError = trainError + f

         -- reset gradients
         gradParameters:zero()

         local df_do = criterion:backward(output, targets)
         net:backward(inputs, df_do)

         local tmpmax, outputlabels = torch.max(output, 2)
         confusion:batchAdd(outputlabels, targets)

         return f,gradParameters
      end

      local old_params = parameters:clone()

      -- optimize
      opt.optimize(feval, parameters, Config)
      grad_over_param = grad_over_param + torch.norm(parameters - old_params) / torch.norm(old_params)
   end

   -- time taken
   time = sys.clock() - time
   
   -- check if step is ok
   grad_over_param = grad_over_param / #dataset.label
   print('average update percent: '..grad_over_param)

   -- print trainError
   trainError = trainError / #dataset.label
   print('train error: ')
   print(trainError)

   -- print confusion matrix
   print(confusion)
   local trainAccuracy = confusion.totalValid * 100
   confusion:zero()
   

   -- next epoch
   epoch = epoch + 1


   return trainAccuracy, trainError
end

------------------------------------------------------------------------
--                              Testing                               --
------------------------------------------------------------------------

-- test function
function test(dataset, ensembleTest)
   net:evaluate()
   -- local vars
   local testError = 0
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> performance on test/val set:')
   
   if ensembleTest then
      Pred = torch.zeros(#dataset.label, opt.nClass):cuda()
   end

   -- disp progress
   for t = 1, #dataset.label, opt.batchSize do
      -- bound batchSize
      local batchSize
      if t + opt.batchSize - 1 > #dataset.label then
         batchSize = #dataset.label - t + 1
      else
         batchSize = opt.batchSize
      end
      local nMax = 0
      -- record the maximum graph size in the mini-batch
      for i = t, t + batchSize - 1 do
         if dataset.ns[i] > nMax then nMax = dataset.ns[i] end
      end
      -- create the mini-batch
      local inputs = {}
      inputs[1] = torch.zeros(batchSize, nMax, nMax):cuda()
      inputs[2] = torch.zeros(batchSize, nMax, dataset.instance[1][2]:size(2)):cuda()
      local targets = torch.Tensor(batchSize):cuda()
      local batchCount = 0
      for i = t, t + batchSize - 1 do
         batchCount = batchCount + 1
         local A = dataset.instance[i][1]
         local x = dataset.instance[i][2]
         local y = dataset.label[i]
         inputs[1][{{batchCount}, {1, A:size(1)}, {1, A:size(2)}}] = A
         inputs[2][{{batchCount}, {1, x:size(1)}, {1, x:size(2)}}] = x
         targets[batchCount] = y
      end
      
      -- test sample
      local pred = net:forward(inputs)
      if ensembleTest then 
         Pred[{{t, t + batchSize - 1}, {}}] = pred
      end
      local tmpmax, outputlabels = torch.max(pred, 2)
      confusion:batchAdd(outputlabels, targets)

      -- compute error
      err = criterion:forward(pred, targets)
      testError = testError + err

   end

   -- testing error estimation
   testError = testError / #dataset.label
   print('test/val error: '..testError)
   
   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- timing
   time = sys.clock() - time

   if ensembleTest then return Pred end

   return testAccuracy, testError
 end

------------------------------------------------------------------------
--                           Main Program                             --
------------------------------------------------------------------------

-- prepare model and data
opt = commandLine()
cutorch.setDevice(opt.gpu)
load_data(opt)
create_model(opt)

-- optimization configurations
if opt.optimization == 'SGD' then
   Config = Config or {learningRate = opt.learningRate,
                       weightDecay = opt.l2reg,
                       momentum = opt.momentum,
                       learningRateDecay = opt.decay_lr}
elseif opt.optimization == 'ADAM' then
   Config = Config or {learningRate = opt.learningRate,  
                       weightDecay = opt.l2reg}
elseif opt.optimization == 'RMSPROP' then
   Config = Config or {learningRate = opt.learningRate,  
                       weightDecay = opt.l2reg}
else
   error('unknown optimization method')
end

-- retrieve parameters and gradients
parameters,gradParameters = net:getParameters()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(opt.nClass)

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, opt.dataName, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, opt.dataName, 'error.log'   ))

-- training and testing
valAcc = 0
valErr = 0
bestValAcc = 0
bestTrainAcc = 0
bestValErr = math.huge
bestIter = 0
bestTestAcc = 0
maxIter = opt.maxEpoch

counter = 0  -- to count how many rounds hasn't the validation error decreased
counter2 = 0 -- to count the ensemble nets

-- show network structure and define some layers for debugging
if opt.debug then
   print(net)
   modus = net.modules
   if opt.originalFeature then 
      modu1 = modus[1].modules[1].modules[2] 
      gc0 = modus[1].modules[1].modules[1]
   else
      modu1 = modus
   end
   gc1 = modu1.modules[1]
   gc1n = modu1.modules[2]
   gc2 = modu1.modules[3].modules[2].modules[1]
   gc2n = modu1.modules[3].modules[2].modules[2]
   gc3 = modu1.modules[3].modules[2].modules[3].modules[2].modules[1]
   gc3n = modu1.modules[3].modules[2].modules[3].modules[2].modules[2]
   gc4 = modu1.modules[3].modules[2].modules[3].modules[2].modules[3].modules[2].modules[1]
   gc4n = modu1.modules[3].modules[2].modules[3].modules[2].modules[3].modules[2].modules[2]
   gc5 = modu1.modules[3].modules[2].modules[3].modules[2].modules[3].modules[2].modules[3].modules[2].modules[1]
   gc5n = modu1.modules[3].modules[2].modules[3].modules[2].modules[3].modules[2].modules[3].modules[2].modules[2]
   ft1 = modus[2]
   jt1 = modus[3]
   sp1 = modus[4]
end


for iter = 1, maxIter do
   print('<<' .. opt.dataName .. '>>')

   -- train
   trainAcc, trainErr = train(trainset)
   
   -- debug, show intermediate layers' parameters
   if opt.debug and iter > 3 then
      --print(gc1n.output[1])
      --print(gc1n.output[2])
      --print(gc2n.output[2])
      --print(gc3n.output[2])
      --print(gc4n.output[2])
      --print(gc1.weight)
      --print(gc1.bias)
      --print(gc1.gradWeight)
      --print(gc2)
      --print(gc2.weight)
      --print(gc2.bias)
      --print(gc2.gradWeight)
      --print(gc3.weight)
      --print(gc3.gradWeight)
      --print(gc3n.gradInput)
      --print(gc4n.output[2])
      print(gc5.weight)
      print(gc5.gradWeight)
      --print(gc5.output[2])
      --print(gc5n.output[2])
      print(sp1.output)
      --print('gradInput of ft1, jt1, sp1')
      --print(ft1.gradInput)
      --print(jt1.gradInput)
      --print(sp1.gradInput)
      debug.debug()
   end

   -- if using validation, test on valset
   if opt.valRatio ~= 0 then
      valAcc, valErr = test(valset)
   end

   timer = torch.Timer()
   -- test on testset
   testAcc, testErr = test(testset)
   print('Time for test dataset: ' .. timer:time().real .. ' seconds')

   counter = counter + 1

   -- if using validation set, update bestNet according to validation error
   if opt.valRatio ~= 0 then
      if valErr < bestValErr then
         counter = 0
         bestValErr = valErr
         bestValAcc = valAcc
         bestIter = iter
         bestTrainAcc = trainAcc
         bestTestAcc = testAcc
         -- save/log current net
         filename = paths.concat(opt.save, opt.dataName, 'bestNet.t7')
         os.execute('mkdir -p ' .. paths.dirname(filename))
         print('<trainer> saving network to '..filename)
         torch.save(filename, net)
      end
      -- if validation error hasn't decreased for many rounds, half the learningRate or earlyStop
      if counter == 20 and opt.halfLR then
         Config.learningRate = Config.learningRate / 2
      elseif counter == 20 and opt.earlyStop then
         break
      end
   end

   -- if using ensemble, save intermediate nets when iter == n * opt.ensemble
   if opt.ensemble ~= 0 then
      if iter % opt.ensemble == 0 then
         counter2 = counter2 + 1
         -- save/log intermediate nets for ensemble
         filename2 = paths.concat(opt.save, opt.dataName, 'interNet'..tostring(counter2)..'.t7')
         os.execute('mkdir -p ' .. paths.dirname(filename2))
         print('<trainer> saving intermediate network to '..filename2)
         torch.save(filename2, net)
      end
   end

      
   -- update logger
   accLogger:add{['% train accuracy'] = trainAcc, ['% val accuracy'] = valAcc, ['% test accuracy'] = testAcc}
   errLogger:add{['% train error']    = trainErr, ['% val error']    = valErr, ['% test error']    = testErr}

   -- plot logger
   accLogger:style{['% train accuracy'] = '-', ['% val accuracy'] = '-', [' test accuracy'] = '.'}
   errLogger:style{['% train error']    = '-', ['% val error']    = '-', [' test error']    = '.'}
   
end


-- After running all epochs --
-- see the performance on testset after all epochs
print('Performance on test set after all epochs: ')
print('train Acc: '..trainAcc..' val Acc: '..valAcc..' test Acc: '..testAcc)

-- if using validation set, load bestNet and see its performance on testset
if opt.valRatio ~= 0 then
   net =  torch.load(filename)
   test(testset)
   print('Best Validation Acc achieved at the '..bestIter..' th iteration:')
   print('train Acc: '..bestTrainAcc..' val Acc: '..bestValAcc..' test Acc: '..bestTestAcc)
   print('The k used is: '..tostring(opt.k))
end

-- if using ensemble, load each interNet and calculate their ensemble prediction performance
if opt.ensemble ~= 0 then
   local Predictions = torch.zeros(#testset.label, opt.nClass):cuda()
   for i = 1, counter2 do
      -- load intermediate nets
      filename2 = paths.concat(opt.save, opt.dataName, 'interNet'..tostring(i)..'.t7')
      net =  torch.load(filename2)
      local Preds = test(testset, true)
      Predictions = Predictions + Preds
   end
   local tmpmax, outputlabels = torch.max(Pred, 2)
   --outputlabels:typeAs(testset.label)
   local tensorTestLabel = torch.Tensor(testset.label):cuda()
   confusion:batchAdd(outputlabels, tensorTestLabel)
   print('The ensemble network performance is:')
   print(confusion)
   ensAcc = confusion.totalValid * 100
   confusion:zero()
end


-- update repeatSave, append current results to trainAcc/testAcc
if opt.repeatSave == true then
   tmp = io.open(paths.concat(opt.save, opt.dataName, 'trainAcc'), 'a');
   io.output(tmp)
   if opt.valRatio == 0 then
      io.write(trainAcc, '\n')
   else
      io.write(bestTrainAcc, '\n')
   end
   io.close()
   tmp = io.open(paths.concat(opt.save, opt.dataName, 'testAcc'), 'a');
   io.output(tmp)
   if opt.valRatio == 0 then
      if opt.ensemble ~= 0 then 
         io.write(ensAcc, '\n')
      else
         io.write(testAcc, '\n')
      end
   else
      io.write(bestTestAcc, '\n')
   end
   io.close()
end

-- plot the accuracy and error curves of this run
accLogger:plot()
errLogger:plot()

