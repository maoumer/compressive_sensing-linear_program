import numpy as np
import cvxpy as cp
from scipy.optimize import linprog


# Basis pursuit optimization solver as linear program
def basis_pursuit(A, b, verbosity=True):
    m, n = A.shape[0], A.shape[1]
    eye = np.eye(n)

    obj_c = np.concatenate([np.zeros(n), np.ones(n)]) #c from c^T x

    lhs_ineq = np.concatenate(
        [
            np.concatenate([eye, -eye], axis=1),
            np.concatenate([-eye, -eye], axis=1),
            np.concatenate([0*eye, -eye], axis=1),
        ], axis=0,)
    rhs_ineq = np.zeros(3 * n)

    lhs_eq = np.concatenate([A, np.zeros((m, n))], axis=1)
    rhs_eq = b

    # added in the inequalities
    # bnd = [
    #     *((None, None) for _ in range(n)),
    #     *((0, None) for _ in range(n)),
    # ]

    #scipy optimize linprog - slow
    res = linprog(
        c=obj_c,
        A_ub=lhs_ineq,
        b_ub=rhs_ineq,
        A_eq=lhs_eq,
        b_eq=rhs_eq, #bounds=bnd,
        method="highs", #highs, highs-ipm, interior-point, simplex, revised simplex
        options={'disp': verbosity})
    return (res.x[:n], res.con, res.status, {'xnorm1': [res.fun]})

    # cvxpy solver
    # x = cp.Variable(2*n)
    # objective = cp.Minimize(obj_c @ x)
    # constraints = [lhs_ineq@x <= rhs_ineq, lhs_eq@x == rhs_eq] #, x[n:] >= 0
    # prob = cp.Problem(objective, constraints) 
    # # The optimal objective value is returned by `prob.solve()`.
    # result = prob.solve(verbose=verbosity)
    # return x.value[:n], "","", {'xnorm1': [result]}


# Basis pursuit denoising optimization solver as linear program
def basis_pursuit_denoising(A, b, sigma, verbosity=True):
    # n -> cols in A
    # m -> rows in B
    m, n = A.shape[0], A.shape[1] 

    eye_n = np.eye(n)
    eye_m = np.eye(m)

    obj_c = np.concatenate([np.zeros(n), np.ones(n), np.zeros(m)])

    lhs_ineq = np.concatenate(
        [
            np.concatenate([eye_n, -eye_n, np.zeros((n, m))], axis=1),
            np.concatenate([-eye_n, -eye_n, np.zeros((n, m))], axis=1),
            np.concatenate([A, np.zeros((m, n)), -eye_m], axis=1),
            np.concatenate([-A, np.zeros((m, n)), -eye_m], axis=1),
            np.concatenate([np.zeros((1, n)), np.zeros((1, n)), np.ones((1, m)),], axis=1,),
            np.concatenate([np.zeros((n, n)), -eye_n, np.zeros((n, m))], axis=1),
            np.concatenate([np.zeros((m, n)), np.zeros((m, n)), -eye_m], axis=1),
        ], axis=0,)

    rhs_ineq = np.concatenate([np.zeros(2 * n), b, -b, [sigma], np.zeros(n + m)])

    res = linprog(
        c=obj_c,
        A_ub=lhs_ineq,
        b_ub=rhs_ineq,
        method="highs", #highs, highs-ipm, interior-point, simplex, revised simplex
        options={'disp': verbosity})
    
    return (res.x[:n], res.con, res.status, {'xnorm1': [res.fun]})

    # cvxpy solver
    # x = cp.Variable(2*n + m)
    # objective = cp.Minimize(obj_c @ x)
    # constraints = [lhs_ineq@x <= rhs_ineq]#, x[n:] >= 0
    # prob = cp.Problem(objective, constraints) 
    # # The optimal objective value is returned by `prob.solve()`.
    # result = prob.solve(verbose=verbosity)
    # return x.value[:n], "","", {'xnorm1': [result]}



def lasso(A, b, tau, verbosity=True):
    # n -> cols in A
    # m -> rows in B
    m, n = A.shape[0], A.shape[1] 

    eye_n = np.eye(n)
    eye_m = np.eye(m)

    obj_c = np.concatenate([np.zeros(n), np.ones(m), np.zeros(n)])

    lhs_ineq = np.concatenate(
        [
            np.concatenate([A, -eye_m, np.zeros((m, n))], axis=1),
            np.concatenate([-A, -eye_m, np.zeros((m, n))], axis=1),
            np.concatenate([eye_n, np.zeros((n, m)), -eye_n], axis=1,),
            np.concatenate([-eye_n, np.zeros((n, m)), -eye_n], axis=1,),
            np.concatenate([np.zeros((1, n)),np.zeros((1, m)),np.ones((1, n))],axis=1,),
            np.concatenate([np.zeros((m, n)), -eye_m, np.zeros((m, n))], axis=1),
            np.concatenate([np.zeros((n, n)), np.zeros((n, m)), -eye_n], axis=1),
        ],
        axis=0,
    )

    rhs_ineq = np.concatenate([b, -b, np.zeros(2 * n), [tau], np.zeros(n + m)],)

    res = linprog(
        c=obj_c,
        A_ub=lhs_ineq,
        b_ub=rhs_ineq,
        method="highs", #highs, highs-ipm, interior-point, simplex, revised simplex
        options={'disp': verbosity})
    
    return (res.x[:n], res.con, res.status, {'xnorm1': [res.fun]})

    # # cvxpy solver
    # x = cp.Variable(2*n + m)
    # objective = cp.Minimize(obj_c @ x)
    # constraints = [lhs_ineq@x <= rhs_ineq]#, x[n:] >= 0
    # prob = cp.Problem(objective, constraints) 
    # # The optimal objective value is returned by `prob.solve()`.
    # result = prob.solve(verbose=verbosity)
    # return x.value[:n],"","", {'xnorm1': [result]}
