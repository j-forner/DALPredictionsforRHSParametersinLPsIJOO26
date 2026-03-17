import numpy as np
import gurobipy as gp
from gurobipy import GRB
from OptProblems import opt
import einops
import torch
import torch.nn as nn

status_dict = {
            2: 'OPTIMAL',
            3: 'INFEASIBLE',
            4: 'INF_OR_UNBD',
            5: 'UNBOUNDED',
            6: 'CUTOFF',
            7: 'ITERATION_LIMIT',
            8: 'NODE_LIMIT',
            9: 'TIME_LIMIT',
            10: 'SOLUTION_LIMIT',
            11: 'INTERRUPTED',
            12: 'NUMERIC',
            13: 'SUBOPTIMAL'
        }

class syn_solver(opt.optSolver):
    def __init__(self, modelSense = opt.MINIMIZE):
        super().__init__(modelSense)
        self.modelSense = modelSense
    
    def solve(self, c, A, b):
        model = gp.Model('synthetic')
        model.params.OutputFlag = 0
        x = model.addMVar(c.shape[0], lb = 0, name = 'x')
        model.setObjective(c @ x, self.modelSense)
        model.addConstr(A @ x >= b)
        model.update()
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return model.X
        else:
            return None
    
    def violation(self, param_constraints, sol, all_comparisons = False):
        A, b = param_constraints
        if all_comparisons:
            first_exp = einops.rearrange(A, 'batch dim num_vars -> batch 1 dim num_vars')
            second_exp = einops.rearrange(sol, 'n num_vars -> 1 n 1 num_vars')

            Ax = torch.einsum('bnic, bnic -> bni', first_exp, second_exp)
            out = einops.rearrange(b, 'batch dim -> batch 1 dim') - Ax
            out = einops.rearrange(out, 'batch n dim -> (batch n) dim')
            return out
        else:
            Ax = einops.reduce(A * sol[:, None], 'b d v -> b d', 'sum')
            violation = b - Ax
            return violation

    def constraint_wise_feasibility(self, param_constraints, sol, all_comparisons = False):
        out = self.violation(param_constraints, sol, all_comparisons = all_comparisons)
        return torch.where(out <= 0, torch.zeros_like(out), torch.ones_like(out))