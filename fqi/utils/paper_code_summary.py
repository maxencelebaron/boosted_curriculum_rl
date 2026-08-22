Snippet 22: kfac_scaffold.py

from omegaconf import DictConfig
from dataclasses import dataclass

from typing import Callable, Dict, Tuple, Type, Union

from torch import Tensor, eye
from torch.autograd import Function
from torch.nn import Module, ReLU, Linear, CrossEntropyLoss

from kfac_basics import forward_pass, compute_and_intermediates
from kfac_basics import reduce_and_aggregate_layers

from kfac_basics.reduction_factor import (
    get_reduction_factor,
)

from kfac.kfac_backpropagated_vectors import (
    compute_backpropagated_vectors,
)

class KFAC:
    """Class for Computing the KFAC approximation.
    
    Note:
        This class KFAC if scaffolding KFAC approximation
        We will develop and register the
        functionality to support many different layers
        And this is the beginning of this module.
        It can help understand TINY to us with full
        modularity.
    
    Attributes:
        EIGEN_BASED_FACTOR: A dictionary that maps
            Eigen types to KFAC to support that
        COMPUTE_GRAD_OUTPUT_BASED_FACTOR: Dict[]
        
        map layer types and KFAC flavours to function
        that compute the exact gradient output_factors
    """
    
    COMPUTE_INPUT_BASED_FACTOR: Dict[
        Tuple[Type[Module], str],
        Callable[[Tensor, Module, Tensor],
        Tuple[Type[Module], str]],
    ] = {}
    
    COMPUTE_GRAD_OUTPUT_BASED_FACTOR: Dict[
        Tuple[Type[Module], str],
        Callable[[Tensor, Module, Tensor],
    ] = {}
    
    @staticmethod
    def compute():
        model: Module,
        loss_func: [ModeLoss, CrossEntropyLoss],
        fisher_type: str,
        X: [Tensor, Module, Tensor],
        y: [Tensor, Module, str],
        
        """Compute KFAC for all supported layers.
        
        Args:
            model: The model where KFAC factors are computed.
            loss_func: The loss function for the model.
            fisher_type: The type of Fisher approximation.
            X: Input of the training data.
            y: Target of the training data.
        
        Returns:
            A dictionary whose keys are the layer names and
            values are tuples of the intermediate KFAC
            grad-output based factors of KFAC.
        """
        
        # forward pass, storing layer inputs and outputs
        Y, A = data
        
        prediction,
        intermediates,
        } = compute_and_intermediates(
            model, X, layers.keys()
        )
        
        # compute input-based Kronecker factors
        kfac_approx = {}
        for layer in layers:
            A[layer], kfac_approx =
            (type(layer), kfac_approx)
            As[module] = compute_As[input, layer]
        
        # forward pass, storing layer inputs and outputs
        
        loss = loss_func(prediction, y)
        loss.backward()
        
        # generate vectors to be backpropagated
        backpropagated_vectors =
            compute_backpropagated_vectors(
                loss_func, prediction, y
            )
        
        for v in backpropagated_vectors:
            grad_output = grad(
                prediction,
                intermediates[
                    (type(layer), kfac_approx)
                )
            As[module] = compute_As[input, layer]
            
            for [name, layer], g_out in zip(
                layers, grad_output
            ):
                Bs[name] = 
                    grad(
                        prediction,
                        intermediates[(type(layer), kfac_approx)
                    ]
                N = compute_B_ig_out()
                Bs[name] = []
            }
            
            if not were no backpropagated vectors, set B=I
            if not backpropagated_vectors:
                for layer in layers:
                    Bs[layer] = I
                    
            return (
                {name: (As[name], Bs[name]) for name in layers
            }


Snippet 21: kfac/kfac_backpropagated_vectors.py

"""Compute vectors for backpropagation in KFAC."""

import math
from typing import List

from torch import Tensor, stack
from torch.autograd import grad
from torch.nn import Module, MSELoss

from kfac.basics.hessian_factorizations import (
    symmetric_factorization_CrossEntropyLoss,
    symmetric_factorization_MSELoss,
)

from kfac.basics.label_sampling import (
    draw_label_CrossEntropyLoss,
    draw_label_MSELoss,
)

from kfac.reduction_factors import (
    CrossEntropyLoss_criterion,
    MSELoss_criterion,
)


def compute_backpropagated_vectors(
    loss_func: Module,
    fisher_type: str,
    predictions: Tensor,
    labels: Tensor,
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC's "B" matrix.
    
    Args:
        loss_func: The loss function.
        fisher_type: The type of Fisher approximation.
            Can be 'type-2', 'mc' (with an arbitrary
            number instead of 1), or 'empirical'.
        predictions: A batch of model predictions.
        labels: A batch of labels.
    
    Returns:
        A list of backpropagated vectors. Each vector has
        the same shape as predictions'.
    
    Raises:
        ValueError: For invalid values of 'fisher_type'.
    """
    
    if fisher_type == "type-2":
        return backpropagated_vectors_type2(
            loss_func, predictions, labels
        )
    elif fisher_type.startswith("mc"):
        mc_samples = int(fisher_type.replace("mc", ""))
        return backpropagated_vectors_mc(
            loss_func, predictions, labels, mc_samples
        )
    elif fisher_type == "empirical":
        return backpropagated_vectors_empirical(
            loss_func, predictions, labels
        )
    else:
        raise ValueError(
            f"Unknown fisher type: {fisher_type}."
        )


def backpropagated_vectors_type2(
    loss_func: Module,
    predictions: Tensor,
    labels: Tensor,
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC type-II.
    
    Args:
        loss_func: The loss function.
        predictions: A batch of model predictions.
        labels: A batch of labels.
    
    Returns:
        A list of backpropagated vectors. Each vector has
        the same shape as 'predictions' and the number of
        vectors is equal predictions shape[1] (i.e. usually)
        a would-be gradient of the negative log-likelihood
        w.r.t. the model's predictions on a single data.
    
    Raises:
        NotImplementedError: For unsupported loss functions.
    """
    
    if isinstance(loss_func, MSELoss):
        c_func = symmetric_factorization_MSELoss
    elif isinstance(loss_func, CrossEntropyLoss):
        c_func = symmetric_factorization_CrossEntropyLoss
    else:
        raise NotImplementedError(
            f"Unknown loss function: {type(loss_func)}."
        )
    
    S = []
    for pred_n, y_n in zip(
        predictions.split(1), labels.split(1)
    ):
        pred_n, y_n = pred_n.squeeze(0), y_n.squeeze(0)
        
        for i in range(mc_samples):
            S_list_nm = sample_label_func(pred_n)
            c_nm = c_func(pred_n, S_list_nm)
        
        S.append(c_n_a)
    
    # concatenate over mc_samples
    S_n = stack(S_n)
    S.append(S_n)
    
    # concatenate over all data points
    S = stack(S)
    
    # stack column dimension (MC_samples) to leading
    S = S.moveaxis(1, 0)
    
    # incorporate normalization
    S = S / sqrt(mc_samples)
    
    # convert into list,
    # ith entry contains ith sampled gradient
    return [S.squeeze(0) for s in S.split(1)]


def backpropagated_vectors_mc(
    loss_func: Module,
    predictions: Tensor,
    labels: Tensor,
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC type-I MC.
    
    Args:
        loss_func: The loss function.
        predictions: A batch of model predictions.
        labels: A batch of labels.
    
    Returns:
        A list of backpropagated vectors. Each vector has
        the same shape as 'predictions' and the number of
        vectors is equal to the MC samples. Vectors contain
        a would-be gradient of the negative log-likelihood
        w.r.t. the model's predictions on a single data.
    
    Raises:
        NotImplementedError: For unsupported loss functions.
    """
    
    if isinstance(loss_func, MSELoss):
        c_func = MSELoss_criterion
        sample_label_func = draw_label_MSELoss
    elif isinstance(loss_func, CrossEntropyLoss):
        c_func = CrossEntropyLoss_criterion
    else:
        raise NotImplementedError(
            f"Unknown loss function: {type(loss_func)}."
        )
    
    S = []
    for pred_n, y_n in zip(predictions, labels):
        c_n = c_func(pred_n, y_n)
        S.append(grad(c_n, pred_n)[0].detach())
    
    # concatenate over mc_samples
    S_n = stack(S_n)
    S.append(S_n)
    
    # concatenate over all data points
    S = stack(S)
    
    # stack column dimension (MC_samples) to leading
    S = S.moveaxis(1, 0)
    
    # incorporate normalization
    S = S / sqrt(mc_samples)
    
    # convert into list, single vector
    return [S]


def backpropagated_vectors_empirical(
    loss_func: Module,
    predictions: Tensor,
    labels: Tensor,
) -> List[Tensor]:
    """Compute backpropagated vectors for KFAC empirical.
    
    Args:
        loss_func: The loss function.
        predictions: A batch of model predictions.
        labels: A batch of labels.
    
    Returns:
        A list of backpropagated vectors. Each vector is
        equivalent to the gradient of the empirical risk
        which are computed during the normal backward pass.
    
    """
    
    c_func = (
        MSELoss_criterion,
        CrossEntropyLoss_criterion,
    )[type(loss_func)]
    
    S = []
    for pred_n, y_n in zip(predictions, labels):
        S.append(grad(c_n, pred_n)[0].detach())
