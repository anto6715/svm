"""

start with:
mpirun -np number_agents python svm_dual_decomposition_mpi_diminishing.py number_dataset iteration


"""

from mpi4py import MPI
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import networkx as nx
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# initialize constants
iteration = int(sys.argv[2])
n = 3                                       # dimension
N = size                                    # number of agents
theory=2.9454744645378845                   # centralized value
n_dataset = int(sys.argv[1])                # number of constraints
n_constraints = int(n_dataset / size)       # number of constraint for each agent

#H is the matrix of quadratic form 
H = np.matrix('1, 0, 0;0, 1, 0; 0 ,0 ,0')

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





# Each agents read his set of neighbor from adjacency matrix
Set_ni = []
Adj = np.loadtxt("adj_matrix")[:][rank]  # only one row is read from an agent
for j in range(0, size):
    if Adj[j] == 1 and j != rank:
        Set_ni.append(j)



# initialize variables

l = np.zeros(shape=(iteration, 1, n))
lambd = np.zeros(shape=(iteration + 1, 2 * len(Set_ni), n))


x1= np.zeros(shape=(int(n_dataset/2), 1))
x2= np.zeros(shape=(int(n_dataset/2), 1))
y1= np.zeros(shape=(int(n_dataset/2), 1))
y2= np.zeros(shape=(int(n_dataset/2), 1))



x = np.zeros(shape=(iteration, len(Set_ni) + 1, n))
X = np.zeros(shape=(n_constraints, 2))
Y = np.zeros(shape=(n_constraints, 1))
A = np.zeros(shape=(n_constraints, 3))
x_best= np.zeros(shape=(iteration,3))
f_best= np.zeros(shape=(iteration,1))

if rank == 0:
    error = np.zeros(shape=(iteration, 1))
    error_media = np.zeros(shape=(iteration, 1))
    x_media = np.zeros(shape=(iteration, 3))
    temp2=0
    f_media = np.zeros(shape=(iteration, 1))

#each agent get the initial condition of lambda_ji
h=0
for z in Set_ni:
    lambd[0][h][:] = np.loadtxt("lambda")[z][rank]
    h = h + 1
# each agent get the points from dataset
u = 0
for i in range(rank * n_constraints, rank * n_constraints + n_constraints):
    X[u] = np.loadtxt("dataset", usecols=(0, 1))[i]
    Y[u] = np.loadtxt("dataset", usecols=2)[i]
    A[u] = Y[u] * np.array([X[u][0], X[u][1], 1])
    u = u + 1



comm.Barrier()
temp=0
for t in range(0, iteration):
    alpha= 1.168*((0.1/(t+1)**(1/2))) #diminishing stepsize
    # subtract lambda_ji
    for k in range(0, len(Set_ni)):
        l[t][0][:] = l[t][0][:] - lambd[t][k][:]
    # add lambda_ij
    for k in range(len(Set_ni), 2 * len(Set_ni)):
        l[t][0][:] = l[t][0][:] + lambd[t][k][:]

    # minimize function using cvxpy
    w = Variable(n)
    obj = Minimize(0.5 * (quad_form(w, H)) + (l[t][0][:]) * w)
    constraints = [A * w >= 1, w <= 5, w >= -5]  # it is possible to add a box constraint
    prob = Problem(obj, constraints)
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
    else:  # agent 0 compute the x average from all agent
        x_best[t]=x[t][0][:]
        for i in range(1,size):
            req = comm.irecv(source=i, tag=0)
            x_best[t] = x_best[t] + req.wait()     
        x_best[t] = x_best[t]/size
        f_best[t]=(1/2)*(np.dot(x_best[t]*H,x_best[t]))


        # agent 0 compute a temporal average for the state
        temp2 = temp2 + x_best[t]
        x_media[t] = temp2 / (t + 1)
        f_media[t] = (1 / 2) * (np.dot(x_media[t] * H, x_media[t]))

        error[t] = (np.absolute(theory) - np.absolute(f_best[t]))
        error_media[t] = (np.absolute(theory) - np.absolute(f_media[t]))



print("agente:", rank, "X:", x[t][0][:])


#plot
if rank == 0:
    print(x_best[t])
    plt.figure(1)
    plt.subplot(311).set_title("(a) error function")
    plt.semilogy(error, label="error function")
    plt.semilogy(error_media, label="error average function")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=0, borderaxespad=0.)

    plt.subplot(312).set_title("(b) function value")
    plt.axhline(y=theory, color='r', linestyle='-', label="centralized value")
    plt.plot(f_best, label="function value")
    plt.plot(f_media, label="average time function value")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)

    r = x[t][0][:]
    b = r[2]
    a1 = -2
    a2 = 5
    c1 = (-b - (r[0] * a1)) / r[1]
    c2 = (-b - (r[0] * a2)) / r[1]
    plt.subplot(313).set_title("(c) hyperplane of svm")
    plt.plot([a1, a2], [c1, c2], 'ro-')
    for i in range(0,20):
        x1[i] = np.loadtxt("dataset", usecols=0)[i]
        y1[i] = np.loadtxt("dataset", usecols=1)[i]

    plt.scatter(x1, y1, s=10, c='blue', alpha=0.5)
    i=0
    for k in range(20,40):
        x2[i] = np.loadtxt("dataset", usecols=0)[k]
        y2[i] = np.loadtxt("dataset", usecols=1)[k]
        i = i+1
    plt.scatter(x2, y2, s=10, c='green', alpha=0.5)
    plt.show()
