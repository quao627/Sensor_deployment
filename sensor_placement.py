import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cdist

from gurobipy import gp, GRB



"""
    Objective: wrong
    First constriant: where is ck?
"""

Ns = 100
Nt = 1000

# def dist(pt1, pt2):
#     return math.sqrt(pow(pt1[0]-pt2[0], 2) + pow(pt1[1]-pt2[1], 2))

def get_O(P, L):
    """Generates matrix O

    Parameters
    ----------
    P : numpy array (n x 2)
        The matrix for sensor positions
    L : float
        Desired minimum distance between two candidate sensors

    Return
    ------
    O : numpy array (n x n)
        Constraint matrix
    """
    dists = cdist(P, P)
    O = np.where(dists < L, 0, 1)
    return O
    
    

def solve_bip(P, v, lam, CVR, L):
    O = get_O(P, L)
    # set up optimization model
    model = gp.Model("bip")
    
    # add decision variables si
    vS = model.addVars(Ns, vtype=GRB.BINARY, name="si")
    # add decision variables ck
    vC = model.addVars(Nt, vtype=GRB.BINARY, name="ci")
    # add regularization term
    # vD = model.addVars(Ns, vtype=GRB.INTEGER, lb=0, ub=Ns, name="Di")
    
    # set objective
    model.setObjective(gp.quicksum(vS[i] for i in range(Ns)) + lam * gp.quicksum(O[i][j] * vS[i] * vS[j] for i in range(Ns) for j in range(Ns)), GRB.MINIMIZE)
    
    # add constraint for visibility
    model.addConstrs(gp.quicksum(v[i][k] * vS[i] for i in range(Ns)) >= vC[k] for k in range(Nt))
    
    # add constraint for maximum # of sensors
    model.addConstrs(gp.quicksum(v[i][k] * vS[i] for i in range(Ns)) <= Ns * vC[k] for k in range(Nt))
    
    # add constraint for coverage
    model.addConstrs(gp.quicksum(vS[k] for k in range(Nt)) >= Nt * CVR)
    
    # solve the model
    model.optimize()
    
    sensors = {k: v.X for k, v in vS.items()}
    
    return sensors
    
    
    
    
    



if __name__ == "__main__":
    
    solve_bip()
