Deep Graph Convolutional Neural Network (DGCNN)
===============================================

About
-----

A powerful deep neural network toolbox for graph classification, named Deep-Graph-CNN (DGCNN). DGCNN features a propagation-based graph convolution layer to extract vertex features, as well as a novel SortPooling layer which sorts vertex representations instead of summing them up. The sorting enables learning from global graph topology, and retains much more node information than summing. The SortPooling layer supports backpropagation, and sorts vertices without using any preprocessing software such as Nauty, which enables an elegant end-to-end training framework.

For more information, please refer to:
> M. Zhang,  Z. Cui,  M. Neumann,  and Y. Chen,  An End-to-End Deep Learning Architecture for
Graph Classification,  Proc. AAAI Conference on Artificial Intelligence (AAAI-18). [\[Preprint\]](http://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGGNN.pdf)

The DGCNN is written in Torch. MATLAB is required if you want to compare DGCNN with graph kernels.

Run "th main.lua" to have a try of DGCNN!

How to run DGCNN
----------------

The folder "data/" contains the .dat graph datasets read by DGCNN. There is already a "MUTAG.dat" included for demo. Other graph datasets need to be generated manually:

    th utils/generate_torch_graphs.lua

which transforms the .mat graph datasets in "data/raw_data/" into .dat format. The "torch.matio" is required to be installed:

    1# OSX
    brew install homebrew/science/libmatio
     # Ubuntu
    sudo apt-get install libmatio2
    2
    luarocks install --local matio

In case you cannot install "torch.matio", we also provide the converted ".dat" for downloading directly [\[Torch graph datasets\]](https://drive.google.com/open?id=1vx19a8UTfj7vboafaoRtgIFv-dIqvhxl).

To run DGCNN on dataset "DD" (whose maximum node label is 89), with learning rate 1e-5, maximum epoch number 100, under the default train/val/test ratio 0.8/0.1/0.1, just type:

    th main.lua -dataName DD -maxNodeLabel 89 -learningRate 1e-5 -maxEpoch 100

To repeatedly run DGCNN on some dataset using different shuffle orders, you can use the provided run_all.sh script:

    time ./run_all.sh XXX gpu_i

which runs DGCNN on dataset XXX with the ith GPU of your machine for 100 times, using the shuffle orders stored in "data/shuffle".

The main program of DGCNN is in "main.lua", please refer to it for more advanced functions and parameters to play with!

How to compare with graph kernels
---------------------------------

The kernel matrices of the compared graph kernels should be put into "data/kernels". Run: 

    ComputeKernels.m

to compute the kernel matrices of WL, GK and RW. Run:
    
    Compute_PK_kernels.m

to compute the kernel matrices of PK (since the PK software uses a different graph format). We also provide the precomputed kernel matrices [\[Kernel matrices\]](https://drive.google.com/open?id=1TneR7RJtRioFcceiIaP6njeKppVbeFFC).

To run graph kernels, run: compare.m in MATLAB, then the accuracy results of all graph kernels will be reported. At the same time, a fixed data shuffling order will be generated and saved in "data/shuffle" for each dataset (thus their train/val/test splits are fixed), for a fair comparison with DGCNN. These shuffle orders are already included in this toolbox. Change the *rand_start* inside compare.m to generate your own shuffle orders.

Required libraries
------------------

Torch libraries needed: paths, torch, nn, cunn, cutorch, optim. Install them using:

    luarocks install --local libraryName

The main folder contains other required modules: SortPooling, GraphConv, GraphReLU, GraphTanh, GraphSelectTable, GraphConcatTable.

To compare DGCNN with graph kernels, libsvm is required to be installed in "software/libsvm-3.22". A zip of libsvm is included in "software/" already, you can unzip it and compile the mex files according to its documentations.

To compare with graph kernels, the toolbox "graphkernels" and "propagation_kernels-master" have been included in software/ already, which come from the following papers:

> N. Shervashidze, P. Schweitzer, E. J. van Leeuwen, K. Mehlhorn, and K. M. Borgwardt.
Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12:2539-2561, 2011.

> Marion Neumann, Roman Garnett, Christian Bauckhage, Kristian Kersting.
Propagation kernels: efficient graph kernels from propagated information. (2015). Machine Learning. 102 (2), pp. 209-245. 

Reference
---------

If you find the code useful, please cite our paper:

    @inproceedings{zhang2018an,
      title={An End-to-End Deep Learning Architecture for Graph Classification.},
      author={Zhang, Muhan and Cui, Zhicheng and Neumann, Marion and Chen, Yixin},
      booktitle={AAAI},
      year={2018}
    }

Muhan Zhang, muhan@wustl.edu
12/2/2017
