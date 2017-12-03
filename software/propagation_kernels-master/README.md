Propagation Kernels
===================

A MATLAB implementation of the propagation graph kernel for general 
graphs (propagation\_kernel.m) and grid graphs (propagation\_kernel\_grid.m). 
The propagation kernel is described in:

> Marion Neumann, Roman Garnett, Christian Bauckhage, Kristian Kersting.
> Propagation kernels: efficient graph kernels from propagated information. (2015). 
> Machine Learning. 102 (2), pp. 209-245. 

and

> Neumann, M. Patricia, N., Garnett, R., and Kersting, K. Efficient
> Graph Kernels by Randomization. (2013). Machine Learning and
> Knowledge Discovery in Databases: European Conference (ECML/PKDD
> 2012), pp. 378--392.

This implementation supports attributed, labeled and weighted graphs and 
arbitrary user-defined base kernels and transformations as well as 
arbitrary user-defined neighborhoods for the grid version. 

Usage
-----

Add the directory to your MATLAB path and use `help
propagation_kernel` to view the documentation. A simple demo  `run_default_PK.m` 
in the `demo/` directory shows how the setup the parameters and run the kernel computation with defaults for labeled graphs. Further demos show how to run the kernel with other than default settings for partially labeled graphs 
(demo.m), attributed graphs (demo_p2k.m) 
and grid graphs, i.e., graphs of regular neighborhood structure, (demo_grid.m). 
