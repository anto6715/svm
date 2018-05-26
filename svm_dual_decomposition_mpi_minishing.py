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
iteration = int(sys.argv[2])
n = 3                                       #dimensionalità
N = size                                    #numero agenti
theory=2.9454744645378845                   #valore teorico
n_dataset = int(sys.argv[1])                #numero totale vincoli
n_constraints = int(n_dataset / size)       #numero di vincoli per agente
H = np.matrix('1, 0, 0;0, 1, 0; 0 ,0 ,0')   #matrice H della quadratica
H2 = np.matrix('1, 0, 0;0, 1, 0; 0 ,0 ,0')  #matrice per il calcolo della funzioen totale


#agente 0 crea grafo
"""
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
    np.savetxt("adj_matrix", Adj)

comm.Barrier()


"""


# Costrusico vettore degli In-neighbor attraverso matrice di adiacenza
Set_ni = []
Adj = np.loadtxt("adj_matrix")[:][rank]  # preleva solo una riga
for j in range(0, size):
    if Adj[j] == 1 and j != rank:
        Set_ni.append(j)



# inizializzazione variabili

l = np.zeros(shape=(iteration, 1, n))
lambd = np.zeros(shape=(iteration + 1, 2 * len(Set_ni), n))
x = np.zeros(shape=(iteration, len(Set_ni) + 1, n))
prova = np.zeros(shape=(iteration, 1))
error = np.zeros(shape=(iteration, 1))
X = np.zeros(shape=(n_constraints, 2))
Y = np.zeros(shape=(n_constraints, 1))
A = np.zeros(shape=(n_constraints, 3))
x_best= np.zeros(shape=(iteration,3))
f_best= np.zeros(shape=(iteration,1))
h=0
for z in Set_ni:
    lambd[0][h][:] = np.loadtxt("lambda")[z][rank]
    h = h + 1
# distribuzione dataset ad ogni agente
u = 0
for i in range(rank * n_constraints, rank * n_constraints + n_constraints):
    X[u] = np.loadtxt("dataset", usecols=(0, 1))[i]
    Y[u] = np.loadtxt("dataset", usecols=2)[i]
    A[u] = Y[u] * np.array([X[u][0], X[u][1], 1])
    u = u + 1

comm.Barrier()
temp=0
for t in range(0, iteration):
    alpha= 1.168*((0.1/(t+1)**(1/2)))
    # sottraggo lambda ji
    for k in range(0, len(Set_ni)):
        l[t][0][:] = l[t][0][:] - lambd[t][k][:]
    # sommo lambda ij
    for k in range(len(Set_ni), 2 * len(Set_ni)):
        l[t][0][:] = l[t][0][:] + lambd[t][k][:]

    # minimizzazione attraverso cvxpy
    w = Variable(n)
    obj = Minimize(0.5 * (quad_form(w, H)) + (l[t][0][:]) * w)
    constraints = [A * w >= 1, w < 5, w > -5]  # va inserito vincolo Ax<b
    prob = Problem(obj, constraints)
    prob.solve()

    x[t][0][:] = np.transpose(w.value)
    temp= temp +x[t][0][2]
    prova[t] = temp/(t+1)


    # invio e ricezione stato con gli agenti vicini
    j = 1;
    for s in Set_ni:
        send = comm.isend(x[t][0][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        x[t][j][:] = req.wait()
        j = j + 1

    comm.Barrier()

    # quando tutti gli agenti hanno scambiato gli stati aggiornano le lambda_ji
    j = 1
    h = 0
    for k in range(0, len(Set_ni)):
        lambd[t + 1][h][:] = lambd[t][h][:] + alpha * (x[t][j][:] - x[t][0][:])
        h = h + 1
        j = j + 1

    # invio lamnbda ji all'agente j e ricevo lambda ij
    i = 0
    for s in Set_ni:
        send = comm.isend(lambd[t + 1][i][:], dest=s, tag=0)
        send.wait()
        req = comm.irecv(source=s, tag=0)
        lambd[t + 1][h][:] = req.wait()
        h = h + 1
        i = i + 1

    # tutti gli agenti inviano il loro stato all'agente 0 per calcolare l'andamento della funzione
    if rank != 0:
        send = comm.isend(x[t][0][:], dest=0, tag=0)
        send.wait()
    else:
        x_best[t]=x[t][0][:]
        for i in range(1,size):
            req = comm.irecv(source=i, tag=0)
            x_best[t] = x_best[t] + req.wait()      #calcolo x medio tra tutti gli agenti
        x_best[t] = x_best[t]/size
        f_best[t]=(1/2)*(np.dot(x_best[t]*H2,x_best[t]))
        temp= theory -f_best[t]
        if temp!=0:
            if temp<0:
                error[t] = math.log(-temp)     #calcolo errore tra valore terorico e valore della funzione
            else:
                error[t] = math.log(temp)
                #alpha = alpha - 0.00001
print("agente:", rank, "X:", x[t][0][:])


#agente 0 esegue plot dell'errore e dell'iperpiano
if rank == 0:
    print(x_best[t])
    plt.figure(1)
    plt.subplot(311).set_title("error function")
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(error)


    plt.subplot(312).set_title("function value")
    plt.axhline(y=theory, color='r', linestyle='-', label="theory value")
    plt.plot(f_best,label="function value")
    plt.legend(bbox_to_anchor=(0.8, 1), loc=0, borderaxespad=0.)
    r = x[t][0][:]
    b = r[2]
    a1 = -2
    a2 = 5
    c1 = (-b - (r[0] * a1)) / r[1]
    c2 = (-b - (r[0] * a2)) / r[1]
    plt.subplot(313).set_title("hyperplane of svm")
    plt.plot([a1, a2], [c1, c2], 'ro-')

    x1 = np.loadtxt("1", usecols=0)
    y1 = np.loadtxt("1", usecols=1)
    plt.scatter(x1, y1, s=10, c='blue', alpha=0.5)
    x2 = np.loadtxt("-1", usecols=0)
    y2 = np.loadtxt("-1", usecols=1)
    plt.scatter(x2, y2, s=10, c='green', alpha=0.5)
    plt.show()