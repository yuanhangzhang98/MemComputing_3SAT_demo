# Solving 3SAT problems using digital MemComputing machines
This demo implements the equations in [Efficient solution of Boolean satisfiability problems with digital memcomputing](https://www.nature.com/articles/s41598-020-76666-2). The [PyTorch](https://pytorch.org/) library is required. 

Usage: 

`python main.py`

The `main.py` file will generate 1000 3SAT instances with 500 variables each and try to solve them. The 3SAT instances are generated according to the rules in [Hiding Solutions in Random Satisfiability Problems: A Statistical Mechanics Approach](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.188701).

If successful, a histogram of the time steps it takes for each instance to reach the solution will be generated at `results/histogram.png`. The distribution of time-to-solution should follow an inverse Gaussian distribution, which we explained in [this paper](https://arxiv.org/abs/2301.08787). 

This demo takes about 1 minute to run on the Nvidia Titan RTX GPU. You can reduce the batch size if the entire dataset doesn't fit in your memory. 

To solve your own 3SAT problems, you can provide .cnf files and customize the `import_data` function in `dataset.py`. Currently a command line parser is not implemented yet. Please raise an issue if you need one, or have any other questions! 
