from mpi4py import MPI
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import networkx as nx;
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

iteration = int(sys.argv[1])
n = 3
alpha = float(sys.argv[2])


A_tot = np.zeros(shape=(n, n))
B_tot = np.zeros(shape=(1))
fval = np.zeros(shape=(iteration, size))

if rank==0:
    for i in range(0,size):
        A_tot= A_tot+ np.load("A_matrices.npy")[i]
        B_tot = B_tot + np.load("B_matrices.npy")[i]
    x_opt = -np.matmul(np.linalg.inv((A_tot)),(B_tot).T)

x_best= np.zeros(shape=(iteration,3))
f_best= np.zeros(shape=(iteration,1))
if rank == 0:
    while (1):
        # Creo la matrice di adiacenza simmetrica e senza elf edges
        Adj = np.random.randint(2, size=(size, size))
        # print(Adj);

        I_n = np.eye(size, dtype=int)
        # print(I_n)
        Adj = np.bitwise_or(Adj, Adj.transpose())
        Adj = np.bitwise_or(Adj, I_n)
        Adj = Adj - I_n
        # print(Adj);

        G = nx.Graph(Adj)
        if (nx.is_connected(G)):
            print("The graph is connected \n")
            break;

    nx.draw(G)
    plt.show()
    # print(Adj)

    np.savetxt("adj_matrix", Adj)

comm.Barrier()


B = np.load("B_matrices.npy")[rank]
A = np.load("A_matrices.npy")[rank]

# Costrusico vettore degli In-neighbor attraverso matrice di adiacenza
Set_ni = []
Adj = np.loadtxt("adj_matrix")[:][rank]  # preleva solo una riga
for j in range(0, size):
    if Adj[j] == 1 and j != rank:
        Set_ni.append(j)

l = np.zeros(shape=(iteration, 1, n))





B=np.load("B_matrices.npy")[rank]
A = np.load("A_matrices.npy")[rank] # preleva solo una riga




lambd = np.zeros(shape=(iteration + 1, 2 * len(Set_ni), n))
x = np.zeros(shape=(iteration, len(Set_ni) + 1, n))
prova = np.zeros(shape=(iteration, 1))


h=0
for z in Set_ni:
    lambd[0][h][:] = np.loadtxt("lambda")[z][rank]
    h = h + 1
comm.Barrier()

for t in range(0, iteration):
    # sottraggo lambda ji
    for k in range(0, len(Set_ni)):
        l[t][0][:] = l[t][0][:] - lambd[t][k][:]
    # sommo lambda ij
    for k in range(len(Set_ni), 2 * len(Set_ni)):
        l[t][0][:] = l[t][0][:] + lambd[t][k][:]

    # minimizzazione attraverso cvx
    w = Variable(n)
    obj = Minimize(0.5 * (quad_form(w, A)) + (B+l[t][0][:]) * w)
    prob = Problem(obj)
    prob.solve()

    x[t][0][:] = 0.5*np.transpose(w.value)

    # invio e ricezione stato con gli agenti vicini
    j = 1;
    for s in Set_ni:
        send = comm.isend(x[t][0][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        x[t][j][:] = req.wait()
        j = j + 1

    comm.Barrier()

    # quando tutti gli agenti hannos cambiato gli stati aggiornano le lmbda_ij e lambda_ji
    j = 1
    h = 0
    for k in range(0, len(Set_ni)):
        lambd[t + 1][h][:] = lambd[t][h][:] + alpha * (x[t][j][:] - x[t][0][:])
        h = h + 1
        j = j + 1

    # invio lamnbda ji all'agente j
    i = 0
    for s in Set_ni:
        send = comm.isend(lambd[t + 1][i][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        lambd[t + 1][h][:] = req.wait()
        h = h + 1
        i = i + 1

    if rank != 0:
        send = comm.isend(x[t][0][:], dest=0, tag=0)
        send.wait()
    else:
        x_best[t]=x[t][0][:]
        for i in range(1,size):
            req = comm.irecv(source=i, tag=0)
            x_best[t] = x_best[t] + req.wait()
        x_best[t] = x_best[t]/size
        f_best[t]=(1/2)*(np.dot(x_best[t],A_tot).dot(x_best[t])+ B_tot.dot(x_best[t]))


print("agente:", rank, "X:", x[t][0][:])

if rank == 0:
    for t in range(0, iteration):
        for i in range(0, size):
            B = np.load("B_matrices.npy")[i]
            A = np.load("A_matrices.npy")[i]
            fval[t][i] = np.dot(x_best[t], A).dot(x_best[t]) + B.dot(x_best[t])
            f_best[t] = f_best[t] + fval[t][i]
    x_best = - np.linalg.inv((A_tot)).dot((np.transpose(B_tot)))
    #fbest = (x_best* A_tot).dot(x_best) + B_tot.dot(x_best)
    print("valore teorico:", x_best/2)
    plt.plot(f_best)
    plt.show()
