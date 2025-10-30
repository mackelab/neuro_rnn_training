import numpy as np
import jaxley as jx
import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
import optax

def simulate(params, stim, input_weights_mask, net, dt=0.025, levels=3):
    """run simulation given stimuli"""
    input_weights = params[0]["input_weights"]*input_weights_mask
    syn_weights = params[1:]

    data_stimuli = None
    for i, w in zip(range(input_weights.shape[0]), input_weights):
        data_stimuli= net.cell(i).data_stimulate(
            jnp.inner(stim,w), data_stimuli=data_stimuli
        )
    num_timesteps = stim.shape[0]
    checkpoints = [int(np.ceil(num_timesteps ** (1/levels))) for _ in range(levels)]
    v = jx.integrate(
        net,
        delta_t=dt,
        params=syn_weights,
        data_stimuli=data_stimuli,
        solver="bwd_euler",
        checkpoint_lengths=checkpoints,
    )
    return v

def predict(opt_params, stim, input_weights_mask, net, dt=0.025, levels=3):
    """extract prediction (readout units activation)"""
    v = simulate(opt_params, stim, input_weights_mask, net, dt=dt, levels=levels)
    n_classes = stim.shape[1] - 1  # last input is fixation
    return ((v[-n_classes:])).T


def softmax(x, axis=1, eps=1e-8):
    """Numerically stable softmax over classes (axis=0)."""
    x_max = jnp.max(x, axis=axis, keepdims=True)
    e_x = jnp.exp(x - x_max)
    e_x /= jnp.sum(e_x, axis=axis, keepdims=True)
    return jnp.clip(e_x, eps, 1 - eps)


def ce(pred, label, mask):
    """
    Cross-entropy loss for one-hot labels (single trial).
    Args:
        pred:  (T, n_classes)
        label: (T, n_classes)
        mask:  (T,1)
    """
    probs = softmax(pred, axis=1)  # softmax over classes at each time
    log_likelihood = jnp.sum(label * jnp.log(probs), axis=1)  # (T,)
    mask = mask.squeeze(-1)
    loss = -jnp.sum(mask * log_likelihood) / jnp.sum(mask)
    return loss


def accuracy(pred, label, mask):
    """
    Compute accuracy for a single trial.
    Args:
        pred:  (T, n_classes)
        label: (T, n_classes)
        mask:  (T, 1)
    """
    probs = softmax(pred, axis=1)
    pred_class = jnp.argmax(probs, axis=1)   # (T,)
    true_class = jnp.argmax(label, axis=1)   # (T,)

    mask = mask.squeeze(-1)
    correct = (pred_class == true_class) * mask
    acc = jnp.sum(correct) / jnp.sum(mask)
    return acc




# initialise optimizer
def init_opt(opt_params, lr, lr_end):

    # Exponential decay of the learning rate.
    scheduler = optax.exponential_decay(
        init_value=lr,
        transition_steps=1,
        decay_rate=0.995,
        end_value=lr_end)

    # Combining gradient transforms using `optax.chain`.
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        optax.scale(-1.0)
    )
    opt_state = gradient_transform.init(opt_params)
    return opt_state, gradient_transform
