-- Main program of DGCNN.
-- *author: Muhan Zhang, Washington University in St. Louis

require 'paths'
require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'optim'

-- load DGCNN-related modules
folderOfThisFile = path.abspath(debug.getinfo(1).short_src):match("(.*[/\\])")
include(folderOfThisFile..'SortPooling.lua')
include(folderOfThisFile..'GraphConv.lua')
include(folderOfThisFile..'GraphReLU.lua')
include(folderOfThisFile..'GraphTanh.lua')
include(folderOfThisFile..'EdgeDropout.lua')
include(folderOfThisFile..'GraphSelectTable.lua')
include(folderOfThisFile..'GraphConcatTable.lua')

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
   cmd:option('-testAfterAll',    false,         'if true, only perform testing after all epochs are finished; otherwise test every epoch')
   cmd:option('-fixed_shuffle',   'random',      'x_y means using data/shuffle/$dataNamex_y.mat as fixed shuffle indices; otherwise "random" means using random shuffle indices, "original" means using original order of the dataset (no shuffle at first)')
   cmd:option('-ensemble',        0,             'if x~=0, use the intermediate nets every x epochs as an ensemble. Using ensemble needs to set -valRatio 0')
   cmd:option('-multiLabel',      false,         'true when doing multi-label classification, use the multi-label one vs all cross entropy loss')
   -- dataset options
   cmd:option('-dataName',        'MUTAG',       'Specify which dataset to use')
   cmd:option('-nClass',          2,             'Specify # of classes of dataset')
   cmd:option('-trainRatio', 	  .9,            'Specify size of train set')
   cmd:option('-valRatio', 	      0,             'Specify size of validation set. Test set size will be 1 - trainRatio - valRatio')
   cmd:option('-testNumber', 	  0,             'if specified, it will overwrite the above trainRatio and valRatio, and use the last "testNumber" examples in the data as the test set, while splitting the remaining data as train (90%) and validation (10%)')
   cmd:option('-maxNodeLabel',    7,             'Specify maximum node label, required if nodeLabel = oneHot')
   -- graph convolution settings
   cmd:option('-bias',            false,         'Whether to include bias b in A(XW+b)')
   cmd:option('-convMatrix',      'rwAplusI',    'Specify which propagation model to use: symAplusI, AplusI, A, rwAplusI')
   cmd:option('-alpha',           1,             'Specify the relative weight of A to I, i.e., I + alpha * A')
   cmd:option('-nodeLabel',       'oneHot',      'Specify node label encoding schemes: original, allOne, nDegree, oneHot, oneHot+nDegree')
   cmd:option('-originalFeature', false,         'whether to add original node features into GraphConv feature vectors')
   cmd:option('-inputChannel',    0,             'Specify # of input channels of the first GraphConv layer. If nodeLabel = original, then this must be specified; otherwise this will be automatically set.')
   cmd:option('-outputChannels',  '32 32 32 1',  'Specify # of output channels of GraphConv layers')
   cmd:option('-nonlinear',       'tanh',        'Specify which nonlinearity to use between GraphConv: relu, tanh, softmax, no')
   cmd:option('-oneWeight',       false,         'whether to use a fixed weight 1 in GraphConv layers')
   cmd:option('-edgeDropout',     0,             'randomly drop out some edges after each GraphConv')
   -- SortPooling options
   cmd:option('-noSortPooling',   false,         'no SortPooling, performs only pooling without sorting.')
   cmd:option('-sumNodeFeatures', false,         'no SortPooling, direclty sum node features followed by only dense layers.')
   cmd:option('-k',               0.6,           'Specify the integer k (how many nodes to keep) in SortPooling. If you set 0 < k <= 1, then k will be converted to an integer so that k% graphs in the dataset have nodes less than this integer. Set k=1 so that k becomes the maximum node number among all graphs')
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
   cmd:option('-maxEpoch', 	      100,           'maximum # of epochs to train for')
   cmd:option('-save', 	          'result',      'result saving position')
   cmd:option('-dataPos', 	      'data',        'data loading position')
   cmd:option('-gpu', 	          1,             'Specify default GPU')
   cmd:option('-log', 	          false,         'whether to log all screen outputs')
   cmd:option('-printAUC', 	      false,         'whether to print AUC score')
   cmd:option('-repeatSave',      true,          'whether to append final results of each run to a file every time for repeated experiments')

   cmd:text()

   local opt = cmd:parse(arg or {})

   print('Running on '..opt.dataName..'...')

   if opt.fixed_shuffle ~= 'random' and opt.fixed_shuffle ~= 'original' then
      torch.manualSeed(opt.seed) -- fixed seed and fixed shuffle for repeatable experiments
      cutorch.manualSeedAll(opt.seed) 
      matio = require 'matio'
      tmp = matio.load(opt.dataPos..'/shuffle/'..opt.dataName..opt.fixed_shuffle..'.mat')
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
      opt.TCkw[1] = opt.totalOutputChannels
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
   local b = {}  -- branches
   local c = {}  -- one condition of the branch
   for i = 1, opt.nGLayers do
      -- add recurrent units from last to first
      local j = opt.nGLayers - i
      c[j] = nn.Sequential()
      if i == 1 then  -- the last layer
         if opt.oneWeight then  -- if fixing the GraphConv weights to 1, i.e., do not learn weights through backpropagation
            c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias, 1))
         else
            c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias))
         end
         c[j]:add(opt.nonlinear())
         c[j]:add(nn.GraphSelectTable(2))
      else  -- graph convolution layers (j=0, 1, ..., r-1) other than the last layer can have EdgeDropout
         if opt.oneWeight then
            c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias, 1))
         else
            c[j]:add(nn.GraphConv(opc[j], opc[j+1], opt.bias))
         end
         if opt.edgeDropout ~= 0 then  -- edge dropout
            c[j]:add(nn.EdgeDropout(opt.edgeDropout))
         end
         c[j]:add(opt.nonlinear())
         c[j]:add(b[j+1])
      end
      
      if j == 0 then  -- the first GraphConv layer
         if opt.originalFeature then  -- if using original node labels/features
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
   if opt.noSortPooling then  -- if removing SortPooling, i.e., using original orders
      net:add(nn.Padding(2, opt.k))
      net:add(nn.Narrow(2, 1, opt.k))
   elseif opt.sumNodeFeatures then  -- if using summed node features
      net:add(nn.Sum(1, 2)) -- sum all node features as a graph-level feature
   else
      net:add(nn.SortPooling(opt.k)) -- default, use SortPooling
   end

   if opt.sumNodeFeatures == false then
      -- now input becomes a (k * totalOutputChannels) tensor
      net:add(nn.View(-1, opt.k * opt.totalOutputChannels, 1))
      -- 1-D convolution layers
      net:add(nn.TemporalConvolution(1, opt.TCChannels[1], opt.TCkw[1], opt.totalOutputChannels))  -- now k * TCChannels[1]
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
   if not opt.multiLabel then
      net:add(nn.LogSoftMax())
   end
   net = net:cuda()

   -- Criterion
   criterion = nn.ClassNLLCriterion()
   if opt.multiLabel then criterion = nn.MultiLabelSoftMarginCriterion() end
   criterion = criterion:cuda()
   print(net)
end
------------------------------------------------------------------------
--                           Data Loader                              --
------------------------------------------------------------------------

local function load_data(opt)
   dataname = opt.dataName
   local dataset = torch.load(opt.dataPos..'/'..dataname..'.dat')
   local train_ratio = opt.trainRatio
   local validation_ratio = opt.valRatio
   local N = #dataset.label
   local Ntrain = math.ceil(N * train_ratio)
   local Nvalidation = math.ceil(N * (train_ratio + validation_ratio)) - Ntrain
   local Ntest = N - Ntrain - Nvalidation
   if opt.testNumber ~= 0 then
      Ntest = opt.testNumber
      Ntrain = math.ceil((N - Ntest) * 0.9)
      Nvalidation = N - Ntrain - Ntest
   end
   local shuffle_idx = torch.Tensor(N)
   if opt.fixed_shuffle == 'random' then
      shuffle_idx = torch.randperm(N)
   elseif opt.fixed_shuffle == 'original' then
      for j = 1, N do shuffle_idx[j] = j end
   else
      shuffle_idx = opt.shuffle_idx:typeAs(shuffle_idx):resizeAs(shuffle_idx)
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
      trainset.instance[i][1] = trainset.instance[i][1]:type('torch.FloatTensor')
      trainset.instance[i][2] = trainset.instance[i][2]:type('torch.FloatTensor')
      trainset.instance[i][2] = processNodeLabel(trainset.instance[i][2], trainset.instance[i][1])
      trainset.instance[i][1] = convMatrix(trainset.instance[i][1])
      trainset.label[i] = dataset.label[shuffle_idx[i]]
      local tmp = trainset.instance[i][1]:size(1)  -- ns: for recording the sizes of graphs
      trainset.ns[i] = tmp
      Ns[i] = tmp
   end
   for i = Ntrain+1, Ntrain+Nvalidation do
      valset.instance[i - Ntrain] = dataset.instance[shuffle_idx[i]]
      valset.instance[i - Ntrain][1] = valset.instance[i - Ntrain][1]:type('torch.FloatTensor')
      valset.instance[i - Ntrain][2] = valset.instance[i - Ntrain][2]:type('torch.FloatTensor')
      valset.instance[i - Ntrain][2] = processNodeLabel(valset.instance[i - Ntrain][2], valset.instance[i - Ntrain][1])
      valset.instance[i - Ntrain][1] = convMatrix(valset.instance[i - Ntrain][1])
      valset.label[i - Ntrain] = dataset.label[shuffle_idx[i]]
      local tmp = valset.instance[i - Ntrain][1]:size(1)
      valset.ns[i - Ntrain] = tmp
      Ns[i] = tmp
   end
   for i = Ntrain+Nvalidation+1, N do
      testset.instance[i - Ntrain - Nvalidation] = dataset.instance[shuffle_idx[i]]
      testset.instance[i - Ntrain - Nvalidation][1] = testset.instance[i - Ntrain - Nvalidation][1]:type('torch.FloatTensor')
      testset.instance[i - Ntrain - Nvalidation][2] = testset.instance[i - Ntrain - Nvalidation][2]:type('torch.FloatTensor')
      testset.instance[i - Ntrain - Nvalidation][2] = processNodeLabel(testset.instance[i - Ntrain - Nvalidation][2], testset.instance[i - Ntrain - Nvalidation][1])
      testset.instance[i - Ntrain - Nvalidation][1] = convMatrix(testset.instance[i - Ntrain - Nvalidation][1])
      testset.label[i - Ntrain - Nvalidation] = dataset.label[shuffle_idx[i]]
      local tmp = testset.instance[i - Ntrain - Nvalidation][1]:size(1)
      testset.ns[i - Ntrain - Nvalidation] = tmp
      Ns[i] = tmp
   end
   if opt.k <= 1 then
      local tmp = torch.sort(Ns)
      opt.k = tmp[math.ceil(opt.k * N)]
      -- erratum --
      -- In paper, I said "k is set so that 60% graphs have #nodes > k", which should have been "k is set so that 60% graphs have #nodes < k"
   end
   if opt.k < 10 then opt.k = 10 end  -- set a lower bound for k, otherwise making 1D convolution infeasible (a small k may result in a negative number of frames after 1D convolutions)

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
      if opt.multiLabel then targets = torch.Tensor(batchSize, opt.nClass):cuda() end
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

         if opt.multiLabel then
            --[[
            -- setting 1: whenever score > 0.5, predict as positive
            local output_scores = torch.cdiv(torch.exp(output),  (1+torch.exp(output)))
            local output_labels = (torch.sign(output_scores - 0.5) + 1) / 2
            ]]
            -- setting 2: select equal number of positive labels as targets (needs to assume # of labels of each testing data is known)
            n_target_labels = torch.sum(targets, 2)
            output_labels = torch.zeros(targets:size()):cuda()
            for target_i = 1, targets:size(1) do
               _, pred_i = torch.topk(output[target_i], n_target_labels[target_i][1], 1, true)
               pred_i = pred_i:type('torch.LongTensor')
               output_labels[target_i]:indexFill(1, pred_i, 1)
            end
            true_pos = true_pos + torch.sum(torch.cmul(output_labels, targets), 1)
            npos = npos + torch.sum(output_labels, 1)
            ntpfn = ntpfn + torch.sum(targets, 1)
         else
            confusion:batchAdd(outputlabels, targets)
         end

         return f, gradParameters
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
   if opt.multiLabel then
      true_neg = #dataset.label - (npos + ntpfn - true_pos)
      trainAccuracy = torch.mean((true_pos + true_neg) / #dataset.label)
      precision = torch.cdiv(true_pos, npos)
      precision[precision:ne(precision)] = 0  -- % avoid nan values (by convention)
      recall = torch.cdiv(true_pos, ntpfn)
      recall[recall:ne(recall)] = 0
      macro_f1 = torch.cdiv(2 * torch.cmul(precision, recall), (precision + recall))
      macro_f1[macro_f1:ne(macro_f1)] = 0
      macro_f1 = torch.mean(macro_f1)
      print('Macro F1 Score is '..tostring(macro_f1))
      print('Mean Training Accuracy is '..tostring(trainAccuracy))
      true_pos = torch.zeros(opt.nClass):cuda()  -- record the # of true positives in each label
      npos = torch.zeros(opt.nClass):cuda()  -- # of positive predictions
      ntpfn = torch.zeros(opt.nClass):cuda() -- # of true positive and false negative predictions (# of positive examples)
   else
      print(confusion)
      trainAccuracy = confusion.totalValid * 100
      confusion:zero()
   end
   
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

   scores = torch.Tensor(#dataset.label):cuda()
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
      if opt.multiLabel then targets = torch.Tensor(batchSize, opt.nClass):cuda() end
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
      scores[{{t, t + batchSize - 1}}] = pred[{{}, {2}}]
      if ensembleTest then 
         Pred[{{t, t + batchSize - 1}, {}}] = pred
      end
      local tmpmax, outputlabels = torch.max(pred, 2)
      if opt.multiLabel then
         --[[
         -- setting 1: whenever score > 0.5, predict as positive
         local output_scores = torch.cdiv(torch.exp(pred), (1+torch.exp(pred)))
         local output_labels = (torch.sign(output_scores - 0.5) + 1) / 2
         ]]
         -- setting 2: select equal number of positive labels as targets (needs to assume # of labels of each testing data is known)
         n_target_labels = torch.sum(targets, 2)
         output_labels = torch.zeros(targets:size()):cuda()
         for target_i = 1, targets:size(1) do
            _, pred_i = torch.topk(pred[target_i], n_target_labels[target_i][1], 1, true)
            pred_i = pred_i:type('torch.LongTensor')
            output_labels[target_i]:indexFill(1, pred_i, 1)
         end
         
         true_pos = true_pos + torch.sum(torch.cmul(output_labels, targets), 1)
         npos = npos + torch.sum(output_labels, 1)
         ntpfn = ntpfn + torch.sum(targets, 1)
      else
         confusion:batchAdd(outputlabels, targets)
      end

      -- compute error
      err = criterion:forward(pred, targets)
      testError = testError + err

   end

   -- testing error estimation
   testError = testError / #dataset.label
   print('test/val error: '..testError)

   -- save prediction scores to file
   tmp = io.open(paths.concat(opt.save, opt.dataName, 'scores'), 'w');
   for row = 1, scores:size(1) do
      tmp:write(scores[row], '\n')
   end
   tmp:close()
   
   -- print confusion matrix
   if opt.multiLabel then
      true_neg = #dataset.label - (npos + ntpfn - true_pos)
      testAccuracy = torch.mean((true_pos + true_neg) / #dataset.label)
      precision = torch.cdiv(true_pos, npos)
      precision[precision:ne(precision)] = 0  -- % avoid nan values (by convention)
      recall = torch.cdiv(true_pos, ntpfn)
      recall[recall:ne(recall)] = 0
      macro_f1 = torch.cdiv(2 * torch.cmul(precision, recall), (precision + recall))
      macro_f1[macro_f1:ne(macro_f1)] = 0
      macro_f1 = torch.mean(macro_f1)
      print('Macro F1 Score is '..tostring(macro_f1))
      print('Mean Testing/Validation Accuracy is '..tostring(testAccuracy))
      true_pos = torch.zeros(opt.nClass):cuda()  -- record the # of true positives in each label
      npos = torch.zeros(opt.nClass):cuda()  -- # of positive predictions
      ntpfn = torch.zeros(opt.nClass):cuda() -- # of true positive and false negative predictions (# of positive examples)
      testError = -macro_f1 -- let the returned testError record the minus F1 score, which is used in selecting best net on validation data
   else
      print(confusion)
      testAccuracy = confusion.totalValid * 100
      confusion:zero()
   end

   -- timing
   time = sys.clock() - time

   if ensembleTest then return Pred end

   if opt.printAUC then
      metrics = require 'metrics'
      local labelTensor = torch.Tensor(dataset.label):cuda()
      roc_points, thresholds = metrics.roc.points(scores, labelTensor, 1, 2)
      auc = metrics.roc.area(roc_points)
      print(auc)
   end

   return testAccuracy, testError
 end

------------------------------------------------------------------------
--                           Main Program                             --
------------------------------------------------------------------------

torch.setdefaulttensortype('torch.FloatTensor')

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
if opt.multiLabel then
   true_pos = torch.zeros(opt.nClass):cuda()  -- record the # of true positives in each label
   npos = torch.zeros(opt.nClass):cuda()  -- # of positive predictions
   ntpfn = torch.zeros(opt.nClass):cuda() -- # of true positive and false negative predictions (# of positive examples)
else
   confusion = optim.ConfusionMatrix(opt.nClass)
end

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, opt.dataName, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, opt.dataName, 'error.log'   ))

-- training and testing
valAcc = 0
valErr = 0
testAcc = 0
testErr = 0
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

print('The k used in SortPooling is: '..tostring(opt.k))

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
   if opt.valRatio ~= 0 or opt.testNumber ~= 0 then
      valAcc, valErr = test(valset)
   end

   if not opt.testAfterAll or iter == maxIter then
      timer = torch.Timer()
      -- test on testset
      testAcc, testErr = test(testset)
      print('Time for test dataset: ' .. timer:time().real .. ' seconds')
   end

   counter = counter + 1

   -- if using validation set, update bestNet according to validation error
   if opt.valRatio ~= 0 or opt.testNumber ~= 0 then
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
if opt.valRatio ~= 0 or opt.testNumber ~= 0 then
   net =  torch.load(filename)
   -- test on testset
   timer = torch.Timer()
   test(testset)
   print('Final inference time on test set: ' .. timer:time().real .. ' seconds')
   print('Best Validation Acc achieved at the '..bestIter..' th iteration:')
   print('train Acc: '..bestTrainAcc..' val Acc: '..bestValAcc..' test Acc: '..bestTestAcc)
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


if opt.printAUC then
   tmp = io.open(paths.concat(opt.save, opt.dataName, 'finalAUC'), 'w');
   tmp:write(auc, '\n')
   tmp:close()
end


if opt.multiLabel then
   tmp = io.open(paths.concat(opt.save, opt.dataName, 'finalF1'), 'w');
   tmp:write(macro_f1, '\n')
   tmp:close()
end


-- update repeatSave, append current results to trainAcc/testAcc
if opt.repeatSave == true then
   tmp = io.open(paths.concat(opt.save, opt.dataName, 'trainAcc'), 'a');
   io.output(tmp)
   if opt.valRatio == 0 and opt.testNumber == 0 then
      io.write(trainAcc, '\n')
   else
      io.write(bestTrainAcc, '\n')
   end
   io.close()
   tmp = io.open(paths.concat(opt.save, opt.dataName, 'testAcc'), 'a');
   io.output(tmp)
   if opt.valRatio == 0 and opt.testNumber == 0 then
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

