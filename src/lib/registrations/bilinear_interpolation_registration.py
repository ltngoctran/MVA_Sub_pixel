import numpy as np

def optim_bilinear_registration(f,g):
    nx,ny = f.shape
    tf = np.zeros((nx+2,ny+2))
    tf[1:-1,1:-1]= f
    # periodic
    tf[0,:] = tf[nx,:]
    tf[nx+1,:] = tf[1,:]
    tf[:,0] = tf[:,ny]
    tf[:,ny+1] = tf[:,1]

    C1 = (tf[1:-1,0:-2]-tf[1:-1,1:-1]).reshape(-1,1)
    C2 = (tf[0:-2,1:-1]-tf[1:-1,1:-1]).reshape(-1,1)
    C3 = (tf[0:-2,0:-2]-tf[0:-2,1:-1]-tf[1:-1,0:-2]+tf[1:-1,1:-1]).reshape(-1,1)

    X = np.hstack((C1,C2,C3))
    Y = (g-f).reshape(-1,1)
    beta = np.linalg.solve(np.matmul(X.T,X),np.matmul(X.T,Y)).flatten()
    return np.abs(beta[0:-1])