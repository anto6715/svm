import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt


N=40
while(1):
    #Creo la matrice di adiacenza simmetrica e senza elf edges
    Adj = np.random.randint(2, size=(N,N))
    #print(Adj);

    I_n = np.eye(N,dtype=int)
    #print(I_n)
    Adj = np.bitwise_or(Adj,Adj.transpose())
    Adj = np.bitwise_or(Adj,I_n)
    Adj = Adj - I_n
    #print(Adj);

    G = nx.Graph(Adj)
    if(nx.is_connected(G)):
        print("The graph is connected \n")
        break;

nx.draw(G)
plt.show(block=False)
print(Adj)

np.savetxt("adj_matrix",Adj)
