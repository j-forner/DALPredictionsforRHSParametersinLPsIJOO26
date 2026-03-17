import numpy as np
import torch
from torch.utils.data import Dataset
from OptProblems.synthetic.synsolver import syn_solver

class syn_dataset(Dataset):

    def __init__(self, xi, c, A, b):
        self.xi = xi
        self.c = c
        self.A = A
        self.b = b
        self.solver = syn_solver()
        self.sols, self.objs = self._getSols()
        self.penaltyVector = np.ones_like(self.c)
    
    def _getSols(self):
        sols = []
        objs = []
        for i in range(len(self.xi)):
            sol = self.solver.solve(self.c, self.A, self.b[i])
            sols.append(sol)
            objs.append([self.c @ sol])
        return np.array(sols), np.array(objs)
    
    def __len__(self):
        return len(self.xi)
    
    def __getitem__(self, index):
        return (
            torch.FloatTensor(self.xi[index]),
            (torch.FloatTensor(self.A), torch.FloatTensor(self.b[index])),
            torch.FloatTensor(self.c),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
            torch.FloatTensor(self.penaltyVector),
        )