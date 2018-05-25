from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

iteration = 200
n = 2
N = 2
alpha = 0.0568
n_lambda=N*N
x = np.zeros(shape=(iteration,N,n))


lambd = np.zeros(shape=(iteration, n_lambda, 2))
l = np.zeros(shape=(iteration, N, n))
if rank == 0:
    A = np.matrix('2, 0;0,4')
    B = np.array([1, 1])

if rank == 1:
    A = np.matrix('3, 0;0,6')
    B = np.array([2, 2])

for t in range(0,iteration-1):



    i = rank
    if i==0:
        j=1
    else:
        j=0
    k = i * N
    k_temp = k
    k_temp2 = i
    temp = N
    temp2 = N
    # print("agente ", i)
    # calcolo sommatoria ij
    while temp > 0:
        l[t][i][:] = l[t][i][:] + lambd[t][k_temp][:]
        k_temp = k_temp + 1
        temp = temp - 1

    # calcolo sommatoria ji
    while temp2 > 0:
        l[t][i][:] = l[t][i][:] - lambd[t][k_temp2][:]
        k_temp2 = k_temp2 + N
        temp2 = temp2 - 1

    x[t][i][:] = -1/2*np.linalg.inv(A).dot(B + l[t][i][:])
    x[t][i][:]=(x[t][i][:])/2
    print("sono qui")
    if rank==0:

        send = comm.isend(x[t][i][:], dest=1, tag=0)
        send.wait()
        print("send1")
        req = comm.irecv(source=1, tag=0)
        x[t][j][:] =req.wait()
        print("rec1")
    if rank==1:

        send = comm.isend(x[t][i][:], dest=0, tag=0)
        send.wait()
        print("send2")
        req = comm.irecv(source=0, tag=0)
        x[t][j][:]=req.wait()
        print("rec2")

    h = 0
    for i in range(0, N):
        for j in range(0, N):
            lambd[t + 1][h][:] = lambd[t][h][:] + alpha * (x[t][i][:] - x[t][j][:])
            h = h + 1
print("agente", rank, x)