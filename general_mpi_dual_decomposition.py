"""

start with:
mpirun -np number_agents python general_mpi_dual_decomposition.py iteration alpha


"""

from mpi4py import MPI
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import networkx as nx;
import sys
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# initialize constants
iteration = int(sys.argv[1])
n = 3
alpha = float(sys.argv[2])

# initialize variables
A_tot = np.zeros(shape=(n, n))
B_tot = np.zeros(shape=(1))
fval = np.zeros(shape=(iteration, size))
x_best= np.zeros(shape=(iteration,3))
f_best= np.zeros(shape=(iteration,1))

# agent 0 reads all matrix A and vector B to calculate centralized value
if rank==0:
    for i in range(0,size):
        A_tot= A_tot+ np.load("A_matrices.npy")[i]
        B_tot = B_tot + np.load("B_matrices.npy")[i]
    x_opt = -np.matmul(np.linalg.inv((A_tot)),(B_tot).T)
    theory = 0.5 * np.dot(x_opt.T, A_tot).dot(x_opt) + B_tot.dot(x_opt)
    print("Centralized value:",theory)


# agent 0 creates a random Erdos--Rényi o Geometric graph
if rank == 0:
    while (1):
        G=nx.erdos_renyi_graph(size,1)
        #G = nx.soft_random_geometric_graph(size, 1)
        A = nx.to_numpy_matrix(G)
        if (nx.is_connected(G) and np.allclose(A, A.T, 1e-8)):
            print("The graph is connected and symmetric\n")
            break;
    nx.draw(G)
    nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    plt.show()
    np.savetxt("adj_matrix", A)
comm.Barrier()

# each agent reads hs matrix A and vector B
B = np.load("B_matrices.npy")[rank]
A = np.load("A_matrices.npy")[rank]

# Each agents read his set of neighbor from adjacency matrix
Set_ni = []
Adj = np.loadtxt("adj_matrix")[:][rank]  # only one row is read from an agent
for j in range(0, size):
    if Adj[j] == 1 and j != rank:
        Set_ni.append(j)


# initialize variables
l = np.zeros(shape=(iteration, 1, n))
lambd = np.zeros(shape=(iteration + 1, 2 * len(Set_ni), n))
x = np.zeros(shape=(iteration, len(Set_ni) + 1, n))
prova = np.zeros(shape=(iteration, 1))


#each agent get the initial condition of lambda_ji
h=0
for z in Set_ni:
    lambd[0][h][:] = np.loadtxt("lambda")[z][rank]
    h = h + 1
comm.Barrier()


if rank == 0:
    error = np.zeros(shape=(iteration, 1))
    error_media = np.zeros(shape=(iteration, 1))
    x_media = np.zeros(shape=(iteration, 3))
    temp2=0
    f_media = np.zeros(shape=(iteration, 1))


for t in range(0, iteration):
    # subtract lambda_ji
    for k in range(0, len(Set_ni)):
        l[t][0][:] = l[t][0][:] - lambd[t][k][:]
    # add lambda_ij
    for k in range(len(Set_ni), 2 * len(Set_ni)):
        l[t][0][:] = l[t][0][:] + lambd[t][k][:]

    # minimize function using cvxpy
    w = Variable(n)
    obj = Minimize(0.5 * (quad_form(w, A)) + (B+l[t][0][:]) * w)
    prob = Problem(obj)
    prob.solve()



    x[t][0][:] = np.transpose(w.value)

    # agent send his state to neighboors and receive their states
    j = 1;
    for s in Set_ni:
        send = comm.isend(x[t][0][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        x[t][j][:] = req.wait()
        j = j + 1

    comm.Barrier()

    # agent i compute  lambda_ji
    j = 1
    h = 0
    for k in range(0, len(Set_ni)):
        lambd[t + 1][h][:] = lambd[t][h][:] + alpha * (x[t][j][:] - x[t][0][:])
        h = h + 1
        j = j + 1

    # agent i sends lambda_ji to agent j and receives lambda_ij
    i = 0
    for s in Set_ni:
        send = comm.isend(lambd[t + 1][i][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        lambd[t + 1][h][:] = req.wait()
        h = h + 1
        i = i + 1
    # agent 0 get the state from all agents
    if rank != 0:
        send = comm.isend(x[t][0][:], dest=0, tag=0)
        send.wait()
    else:	# agent 0 compute the x average from all agent
        x_best[t]=x[t][0][:]
        for i in range(1,size):
            req = comm.irecv(source=i, tag=0)
            x_best[t] = x_best[t] + req.wait()
        x_best[t] = x_best[t]/size
        f_best[t]=(1/2)*(np.dot(x_best[t],A_tot).dot(x_best[t]))+ B_tot.dot(x_best[t])

        # agent 0 compute a temporal average for the state
        temp2 = temp2 + x_best[t]
        x_media[t] = temp2 / (t + 1)
        f_media[t] = (1/2)*(np.dot(x_media[t],A_tot).dot(x_media[t]))+ B_tot.dot(x_media[t])


        error[t] = (np.absolute(theory) - np.absolute(f_best[t]))
        error_media[t] = (np.absolute(theory) - np.absolute(f_media[t]))


print("agente:", rank, "X:", x[t][0][:])

# plot
if rank == 0:
    print(x_opt)200


    plt.figure(1)
    plt.subplot(211).set_title("(1) error function")
    plt.semilogy(error, label="error function")
    plt.semilogy(error_media, label="error temporal average function")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=0, borderaxespad=0.)

    plt.subplot(212).set_title("(2) function value")
    plt.axhline(y=theory, color='r', linestyle='-', label="centralized value")
    plt.plot(f_best, label="function value")
    plt.plot(f_media, label="function value with temporal average ")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
    plt.show()