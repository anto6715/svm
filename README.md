# IMPLEMENTATION OF A DISTRIBUTED OPTIMIZATION VIA DUAL DECOMPOSITION ALGORITHM IN ORDER TO SOLVE A SVM PROBLEM

-**[general_mpi_dual_decomposition.py](https://github.com/anto6715/svm/blob/master/general_mpi_dual_decomposition.py "general_mpi_dual_decomposition.py")** is an implementations of the support vector machine algorithm solved using traditional dual decomposition method,

-**[svm_dual_decomposition_mpi_diminishing.py](https://github.com/anto6715/svm/blob/master/svm_dual_decomposition_mpi_diminishing.py "svm_dual_decomposition_mpi_diminishing.py")** is an implementations of the support vector machine algorithm solved using mpi

## Requirements

For building and running the application you need:

- [MPI4py](https://mpi4py.readthedocs.io/en/stable/)

## Run
```mpirun -np number_agents python svm_dual_decomposition_mpi.py number_dataset iteration alpha
```
mpirun -np number_agents python svm_dual_decomposition_mpi.py number_dataset iteration alpha

## Description:
We study a distributed multi-agent optimization problem of minimizing the
sum of some convex objective functions. We introduce a decentralized op-
timization algorithm based on the dual decomposition theory and the sub-
gradient method. Firstly, the algorithm is implemented on Matlab in order
to obtain a series of results used to validate the distributed algorithm. Then
we set up a MPI communication protocol with the aim of simulating the
communication between some agents, particularly we use MPI4PY which
combines the MPI functionalities and the programming language Python,
one of the most used languages in machine learning. Afterwards the imple-
mented algorithm is applied to a particular problem about machine learning,
called Support Vector Machine. We study the SVM models, then we apply
the hard margin theory (linear SVM) modifying the optimization problem
and we test the implemented model to a given linearly separable dataset in
order to validate the final result of the project. Finally we use the imple-
mented model to show some interesting charts about the error function, the
trend of the temporal mean of the states and the computed hyperplane or
the behaviour of the algorithm with the application of a penalty function or
a diminishing stepsize, instead of a constant stepsize.



## Authors

* **Antonio Mariani** 
* **Giuseppe Morleo**
* **Riccardo Contino**
* **Andrea Della Monaca**
