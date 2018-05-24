from mpi4py import MPI
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import networkx as nx;
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_dataset = int(sys.argv[1])
n_constraints = int(n_dataset / size)
print(n_constraints)
X = np.zeros(shape=(n_constraints, 2))
Y = np.zeros(shape=(n_constraints, 1))


A= np.zeros(shape=(n_constraints,3))
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

# distribuzione dataset ad ogni agente
u = 0
for i in range(rank * n_constraints, rank * n_constraints + n_constraints):
    X[u] = np.loadtxt("dataset completo", usecols=(0, 1))[i]
    Y[u] = np.loadtxt("dataset completo", usecols=2)[i]
    A[u] = Y[u]*np.array([X[u][0],X[u][1], 1])
    u = u + 1

iteration = int(sys.argv[2])
n = 3
N = size
alpha = float(sys.argv[3])

# Costrusico vettore degli In-neighbor attraverso matrice di adiacenza
Set_ni = []
Adj = np.loadtxt("adj_matrix")[:][rank]  # preleva solo una riga
for j in range(0, size):
    if Adj[j] == 1 and j != rank:
        Set_ni.append(j)

l = np.zeros(shape=(iteration, 1, n))
H = np.matrix('1, 0, 0;0, 1, 0; 0 ,0 ,0.02')
lambd = np.zeros(shape=(iteration + 1, 2 * len(Set_ni), n))
x = np.zeros(shape=(iteration, len(Set_ni) + 1, n))
prova = np.zeros(shape=(iteration, 1))

comm.Barrier()

for t in range(0, iteration):
    # costruisco lambda_i sommando le lamnda_ij e sottraendo lambda_ji
    for k in range(0, len(Set_ni)):
        l[t][0][:] = l[t][0][:] + lambd[t][k][:]
    # sottraggo lambda ji
    for k in range(len(Set_ni), 2 * len(Set_ni)):
        l[t][0][:] = l[t][0][:] - lambd[t][k][:]

    # minimizzazione attraverso cvx
    w = Variable(n)
    obj = Minimize(0.5 * (quad_form(w, H)) + (l[t][0][:]) * w)
    constraints = [A * w >= 1, w < 5, w > -5]  # va inserito vincolo Ax<b
    prob = Problem(obj, constraints)
    prob.solve()

    x[t][0][:] = np.transpose(w.value)
    prova[t] = x[t][0][0]

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
        lambd[t + 1][h][:] = lambd[t][h][:] + alpha * (x[t][0][:] - x[t][j][:])
        h = h + 1
        j = j + 1

    j = 1
    for k in range(0, len(Set_ni)):
        lambd[t + 1][h][:] = lambd[t][h][:] + alpha * (x[t][j][:] - x[t][0][:])
        h = h + 1
        j = j + 1

print("agente:", rank, "X:", x[t][0][:])

if rank == 2:
    plt.figure(1)
    plt.subplot(211)
    plt.plot(prova)
    #plt.show()
    r=x[t][0][:]
    b = r[2]
    a1 = -2
    a2 = 5
    c1 = (-b - (r[0] * a1)) / r[1]
    c2 = (-b - (r[0] * a2)) / r[1]
    #plt.plot
    plt.subplot(212)
    plt.plot([a1, a2], [c1, c2], 'ro-')

    N = 50
    x1 = np.loadtxt("1", usecols=0)
    y1 = np.loadtxt("1", usecols=1)
    plt.scatter(x1, y1, s=10, c='blue', alpha=0.5)
    #plt.plot(prova)
    x2 = np.loadtxt("-1", usecols=0)
    y2 = np.loadtxt("-1", usecols=1)
    plt.scatter(x2, y2, s=10, c='green', alpha=0.5)
    plt.show()

