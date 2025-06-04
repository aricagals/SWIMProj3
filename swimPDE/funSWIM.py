from functools import partial
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from swimnetworks import Linear
from swimpde.ansatz import BasicAnsatz
from swimpde.domain import Domain


# Phisics parameters of the Eikonal Equation
a_ratio = 8.8131
Cv = 75.2629
theta_fiber = 0
theta0 = np.pi/2 - theta_fiber
a = np.array([np.cos(theta0), np.sin(theta0)]).T
b = np.array([np.cos(theta0-np.pi/2), np.sin(theta0-np.pi/2)]).T
D = ( (1/a_ratio)*np.tensordot( a, a,  axes=0) + np.tensordot( b, b,  axes=0)  )

# Forcing term
def forcing(v):
    return -np.ones((v.shape[0],1))


# Evaluation data to test the results on.
n_test_sqrt = 101
x_span = -2, 2
n_dim = 2

x_test = np.linspace(*x_span, num=n_test_sqrt)
y_test = np.linspace(*x_span, num=n_test_sqrt)
xy_test = np.stack(np.meshgrid(x_test, y_test), axis=-1)
xy_test = xy_test.reshape(-1, n_dim)


# Ansartz + Lineaar
def get_model(points, target, n_basis, seed):
    domain = Domain(interior_points=points)
    ansatz = BasicAnsatz(n_basis=n_basis,
                        activation="tanh",
                        random_seed=seed)
    ansatz.fit(domain, target)
    linear = Linear(regularization_scale=1e-12)
    if target is not None:
        weights = np.linalg.lstsq(
            ansatz.transform(points), target, rcond=1e-12
            )[0]
        linear.weights = weights
    linear.biases = np.zeros((1, 1))
    linear.layer_width = 1
    return ansatz, linear

def swim_train (xy_measurement, u_measured):

    # Domain data
    n_col_sqrt = 4

    x_col = np.linspace(*x_span, num=n_col_sqrt)
    y_col = np.linspace(*x_span, num=n_col_sqrt)
    xy_col = np.stack(np.meshgrid(x_col, y_col), axis=-1)
    xy_col = xy_col.reshape(-1, n_dim)

    n_basis = 800
    u_seed = 99

    u_ansatz, u_linear = get_model(xy_measurement, u_measured, n_basis, u_seed)

    forcing_col = forcing(xy_col)
    training_start = time()
    
    u_phi_measured = u_ansatz.transform(xy_measurement)
    u_phi_x = u_ansatz.transform(xy_col, operator="gradient")

    # Least squares for computing u_approx: use updated value of gamma
    # and also stack the true values we know at the measurement points
    matrix_in_u = Cv * np.sqrt(np.einsum("nkd,df,nkf->nk", u_phi_x, D, u_phi_x))
        
    matrix_in_u = np.row_stack([matrix_in_u, u_phi_measured])
    matrix_out_u = np.concatenate([forcing_col, u_measured])

    u_outer_weights = np.linalg.lstsq(matrix_in_u, matrix_out_u, rcond=1e-12)[0]

    u_linear.weights = u_outer_weights

    training_time = time() - training_start
    print(f"Training time: {training_time}")

    u_model = Pipeline([("ansatz", u_ansatz), ("linear", u_linear)])
    u_pred = u_model.transform(xy_test).ravel()

    return u_pred

