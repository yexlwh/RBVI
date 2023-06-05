import numpy as np
from normwish1 import *
def SVDOPPos(gammas,ep):
    # ############################################################################################################
    # print(gammas)
    N, K = gammas.shape
    u_SVD, s_SVD, v_SVD = np.linalg.svd(gammas, full_matrices=True)
    #
    # s_SVD = np.diag(s_SVD)
    D_u = (u_SVD.shape)[0]
    D_s = (s_SVD.shape)[0]
    S = s_SVD + ep
    s_SVD = np.zeros((D_u, D_s))
    s_SVD[:D_s, :D_s] = np.diag(S)
    # print(s_SVD)
    gammas = np.dot(u_SVD, np.dot(s_SVD, v_SVD))
    # print(gammas)
    gammas = (gammas + abs(gammas)) / 2;
    # ######################################################################
    gammas = gammas / repmat(np.sum(gammas, axis=1), K, 1).T;
    return gammas