# Needed libraries
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from functools import partial
from tqdm import tqdm

import numpy as onp

from pyDOE import lhs 

# Random seed setted
jax.config.update("jax_enable_x64", True)
key = jax.random.key(2304)
rngs = nnx.Rngs(2304)
onp.random.seed(2304)
jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(2304)
rngs = nnx.Rngs(key) 
onp.random.seed(2304) 


# Global constants of the problem
N_MESH = 101
X_mesh, Y_mesh = np.meshgrid(np.linspace(-2.0, 2.0, N_MESH), np.linspace(-2.0, 2.0, N_MESH))
L = 2
a_ratio = 8.8131
Cv = 75.2629
theta_fiber = 0
theta0 = np.pi/2 - theta_fiber
a = np.array([np.cos(theta0), np.sin(theta0)]).T
b = np.array([np.cos(theta0-np.pi/2), np.sin(theta0-np.pi/2)]).T
D = ( (1/a_ratio)*np.tensordot( a, a,  axes=0) + np.tensordot( b, b,  axes=0)  )



# collocation points
Ncl = 2000
X_coll = lhs(2,2000)
X_coll = -2.0 + X_coll * 4.0

# border points
Nbd = 100

# Left boundary strip (-2 <= x <= -1.9, y in [-2,2])
Sample_y = lhs(1, Nbd).flatten() * 4.0 - 2.0  # Scala in [-2,2]
Sample_x_left = lhs(1, Nbd).flatten() * 0.1 - 2.0  # Scala in [-2,-1.9]
X_bound_left = jnp.column_stack((Sample_x_left, Sample_y))

# Right boundary strip (1.9 <= x <= 2, y in [-2,2])
Sample_x_right = lhs(1, Nbd).flatten() * 0.1 + 1.9  # Scala in [1.9,2]
X_bound_right = jnp.column_stack((Sample_x_right, Sample_y))

# Bottom boundary strip (-2 <= y <= -1.9, x in [-2,2])
Sample_x = lhs(1, Nbd).flatten() * 4.0 - 2.0  # Scala in [-2,2]
Sample_y_bottom = lhs(1, Nbd).flatten() * 0.1 - 2.0  # Scala in [-2,-1.9]
X_bound_bottom = jnp.column_stack((Sample_x, Sample_y_bottom))

# Top boundary strip (1.9 <= y <= 2, x in [-2,2])
Sample_y_top = lhs(1, Nbd).flatten() * 0.1 + 1.9  # Scala in [1.9,2]
X_bound_top = jnp.column_stack((Sample_x, Sample_y_top))

# Stack all boundary points
X_bound = jnp.vstack([X_bound_left, X_bound_right, X_bound_bottom, X_bound_top])



# MLP definition
from typing import Callable, Optional, Sequence

class CustomMLP(nnx.Module):
    """
    Flexible Multi-Layer Perceptron (MLP) implementation.

    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        hidden_features: Sequence of hidden layer sizes
        dropout_rate: Dropout rate to use in hidden layers
        activation_function: Activation function to use in hidden layers
        final_activation_function: Activation function to use in output layer
        rngs: Random number generators necessary for dropout layers
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rngs: nnx.Rngs,
            hidden_features: Sequence[int],
            dropout_rate: float = 0.0,
            activation_function: Callable = lambda x: x,
            final_activation_function: Optional[Callable] = lambda x: x,
            # final_activation_function: Optional[Callable] = None,
            
    ):

        super().__init__()

        # Store configuration
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.final_activation_function = final_activation_function

        self.linear_layers = []
        self.dropout_layers = []

        layer_params = {"kernel_init":nnx.initializers.glorot_normal(), 
                        "dtype":jnp.float64, 
                        "rngs":rngs}

        current_in_features = in_features
        # Build hidden layers structure
        for hidden_size in self.hidden_features:
            self.linear_layers.append(nnx.Linear(current_in_features, hidden_size, **layer_params))
            # Add dropout layer
            self.dropout_layers.append(nnx.Dropout(self.dropout_rate, rngs=layer_params["rngs"]))
            # Update input size for next layer
            current_in_features = hidden_size

        self.linear_layers.append(nnx.Linear(current_in_features, out_features, **layer_params))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the network.
        Args:
            inputs: Input tensor of shape (..., in_features)
        Returns:
            Output tensor of shape (..., out_features)
        """

        # Process hidden layers
        for linear_layer, dropout_layer in zip(self.linear_layers[:-1], self.dropout_layers):

            # Forward pass through layer
            x = linear_layer(x)
            x = self.activation_function(x)
            x = dropout_layer(x)

        # Process output layer
        x = self.linear_layers[-1](x)
        # Apply final activation if specified
        x = self.final_activation_function(x) if self.final_activation_function is not None else x

        return x
    

@nnx.jit
def squeeze_pinn_output(pinn, x):
    """Squeeze PINN model output."""
    return pinn(x).squeeze()

grad_pinn_scalar = nnx.grad(squeeze_pinn_output, argnums=1)
grad_pinn = nnx.jit(nnx.vmap(grad_pinn_scalar, in_axes=(None, 0)))

# residual computation based on AD
@nnx.jit
def r_pinn(pinn, X):
    """Compute residual function."""
    grad_u = grad_pinn(pinn, X)
    # D_expanded = np.tile(D, (Ncl, 1, 1))
    # grad_D_grad = jnp.einsum("ij, ijk, ik -> i", grad_u, D_expanded, grad_u)
    grad_D_grad = jnp.einsum("ij, ij -> i", grad_u @ D, grad_u)
    
    grad_D_grad = grad_D_grad[:,None]

    # print(grad_D_grad.shape)
    
    return Cv * jnp.sqrt(grad_D_grad) - 1


# Residual loss: PINN residual su collocation points
def loss_r(pinn, r_pinn, X_coll):
    r_pred = r_pinn(pinn, X_coll)
    return jnp.mean(jnp.square(r_pred))

# Misure: errore sui punti misurati
def loss_meas(pinn, X_meas, u_meas):
    u_meas_pred = pinn(X_meas)
    return jnp.mean(jnp.square(u_meas_pred - u_meas))


# Regolarizzazione: minima penalità
def loss_reg(pinn, X_coll):
    return (jnp.min(pinn(X_coll)) - 0.000) ** 2

def loss_reg_smooth(pinn, X):
    # grad_u = grad_pinn(pinn, X)  # Calcola il gradiente
    # return jnp.mean(jnp.square(grad_u))
    return (nnx.relu( - jnp.min(pinn(X)) + 0.015)) ** 2


@partial(nnx.jit, static_argnames=('r_pinn'))
def loss(pinn, r_pinn, X_coll, X_meas, u_meas, X_bound):
    
    mse_r = loss_r(pinn, r_pinn, X_coll)
    mse_meas = loss_meas(pinn, X_meas, u_meas)
    reg = loss_reg(pinn, X_coll)
    reg_smooth = loss_reg_smooth(pinn, X_bound) 

    return 1*mse_meas + 1e-4*mse_r + 5e-1*reg + 1*reg_smooth

@partial(nnx.jit, static_argnames=('r_pinn'))

def train_step(opt, pinn, r_pinn, X_coll, X_meas, u_meas, X_bound):
    grad_fn = nnx.value_and_grad(loss)
    value, grads = grad_fn(pinn, r_pinn, X_coll, X_meas, u_meas, X_bound)
    opt.update(grads)

    return value


# PINN Training
def pinn_net(X_meas, u_meas):
  
  jax.config.update("jax_enable_x64", True)
  key = jax.random.key(2304)
  rngs = nnx.Rngs(2304)
  onp.random.seed(2304)

  pinn = CustomMLP(in_features=2,
                  out_features=1, 
                  hidden_features=[8,16],
                  rngs=rngs,
                  dropout_rate=0.0,
                  activation_function=nnx.tanh, 
                  final_activation_function=nnx.tanh)

  # Adam optimizer
  num_epochs = 40000

  scheduler = optax.exponential_decay(
      init_value=0.003,  # Learning rate iniziale
      transition_steps=2000,  # Dopo quante epoche iniziare il decay
      decay_rate=0.977,
      end_value=0.0001 # Fattore di riduzione
  )

  optimizer = nnx.Optimizer(pinn, optax.adam(learning_rate=scheduler,b1=0.99))

  pbar = tqdm(range(num_epochs), desc="Training", dynamic_ncols=True)
  for iter in pbar:

    loss_value = train_step(optimizer, pinn, r_pinn, X_coll, X_meas, u_meas, X_bound)

  # Definisci i limiti
  x_min, x_max = -2, 2
  y_min, y_max = -2, 2
  num_points = 101  # Risoluzione della griglia

  # Crea la meshgrid
  x = jnp.linspace(x_min, x_max, num_points)
  y = jnp.linspace(y_min, y_max, num_points)
  X, Y = jnp.meshgrid(x, y)

  # Evaluta il PINN su questa griglia
  Z = pinn(jnp.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

  return Z
