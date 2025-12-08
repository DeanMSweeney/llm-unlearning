import torch
import torch.nn.functional as F
from typing import Optional, Literal, Union, Tuple, Dict, List

# Mixin class for Parameter-Efficient Gradient Unlearning (PCGU)
# Implements selective parameter updating based on gradient similarity analysis
class PCGUMixin:
    
    def _top_k_params(self) -> List[float]:
        """
        Selects the top k most influential parameters based on gradient similarity.
        For two-class: Parameters with lower cosine similarity between grad_1 and grad_2 are selected.
        For three-class: Parameters with higher variance across all three gradients are selected.

        Args
        ---------
        param_partition: (param_name, index) pairs denoting the partition of the weights
        grad_1: dict from param_name to gradient tensor
        grad_2: dict from param_name to gradient tensor
        grad_3: optional dict from param_name to gradient tensor (for three-class case)
        """

        # Compute similarity/variance scores for each parameter partition
        scores = []
        for param_name, indices in self.param_partition:
            # Retrieve gradients for the current parameter
            self.grad_1 = self.grads_1[param_name]
            self.grad_2 = self.grads_2[param_name]
            self.grad_3 = self.grads_3[param_name] if self.grads_3 is not None else None

            # If gradients don't exist, assign a large similarity score to exclude this parameter
            if self.grad_1 is None or self.grad_2 is None:
                scores.append(torch.Tensor([5]).squeeze())
                continue

            # If this parameter is partitioned, extract the relevant gradient indices
            if indices is not None:
                self.grad_1 = self.grad_1[indices]
                self.grad_2 = self.grad_2[indices]
                if self.grad_3 is not None:
                    self.grad_3 = self.grad_3[indices]

            # For three-class case: compute variance across gradients
            if self.grad_3 is not None:
                # Stack gradients and compute variance
                # Higher variance means more disagreement across classes
                grad_stack = torch.stack([self.grad_1, self.grad_2, self.grad_3])
                variance = torch.var(grad_stack, dim=0).mean().detach().cpu()
                # Negate variance so we can still use topk with largest=False
                # (we want high variance = low score for consistent selection)
                scores.append(-variance)
            # For two-class case: compute cosine similarity
            else:
                # Lower similarity indicates more conflicting updates (more influential)
                cosine_sim = F.cosine_similarity(self.grad_1, self.grad_2, dim=-1).detach().cpu()
                scores.append(cosine_sim)

        # Select the top k most influential parameters based on similarity scores
        # Stack all similarity scores into a single tensor
        sim_stack = torch.stack(scores)

        # Select the k parameters with the smallest similarity scores (largest=False)
        # For two-class: parameters where grad_1 and grad_2 disagree the most
        # For three-class: parameters with highest variance (most negative score)
        top_k_result = sim_stack.topk(self.k, largest=False, sorted=True)
        target_indices = [ind.item() for ind in top_k_result[1]]
        self.params_to_keep = [self.param_partition[ind] for ind in target_indices]

    def _rewrite_grad(self, grad_1, grad_2, grad_3=None):
        """
        Determines which gradient to use for parameter updates based on the selected strategy.

        Args:
        ---------
            grad_1: Gradient from the first variant (e.g., male or advantaged)
            grad_2: Gradient from the second variant (e.g., female or disadvantaged)
            grad_3: Optional gradient from third variant (e.g., neutral)

        Returns:
            The selected gradient based on self.which_grad strategy
        """
        # For three-class case, minimize variance
        if grad_3 is not None:
            if self.which_grad == "combined":
                # Combine all three gradients to minimize variance across classes
                return -(grad_1 + grad_2 + grad_3)
            else:
                # Default to combined strategy for three-class case
                return -(grad_1 + grad_2 + grad_3)
        # For two-class case, use original strategies
        else:
            # Use only the advantaged/first gradient
            if self.which_grad == "advantaged":
                return grad_1
            # Use the negated disadvantaged/second gradient
            elif self.which_grad == "disadvantaged":
                return -grad_2
            # Use the negated sum of both gradients (neutral/balanced approach)
            elif self.which_grad == "combined":
                return -(grad_1 + grad_2)
            else:
                raise AttributeError("Gradient method not selected")
        
    def update_model_param_grads(
        self,
        model_params_map,
    ):
        """
        Updates the gradients of model parameters based on the selected parameters
        and gradient rewriting strategy.

        Args:
        ---------
            model_params_map: Dictionary mapping parameter names to parameter tensors
        """
        # Zero out all gradients so unselected parameters remain at zero gradient
        self.optimizer.zero_grad(set_to_none=False)

        # If specific parameters were selected (via top-k), only update those
        if self.params_to_keep is not None:
            for param_name, indices in self.params_to_keep:
                param = model_params_map[param_name]
                # If the parameter is not partitioned, update the entire gradient
                if indices is None:
                    grad_3 = self.grads_3[param_name] if self.grads_3 is not None else None
                    new_grad = self._rewrite_grad(grad_1=self.grads_1[param_name], grad_2=self.grads_2[param_name], grad_3=grad_3)
                    param.grad.data.copy_(new_grad.data)
                # If the parameter is partitioned, only update specific indices
                else:
                    grad_3 = self.grads_3[param_name][indices] if self.grads_3 is not None else None
                    new_grad = self._rewrite_grad(grad_1=self.grads_1[param_name][indices], grad_2=self.grads_2[param_name][indices], grad_3=grad_3)
                    param.grad[indices] = new_grad.to(self.device)
        # If no specific parameters selected, update all parameters with valid gradients
        else:
            # Update all parameters that have all required gradients defined
            for param_name, param in model_params_map.items():
                if self.grads_1[param_name] is not None and self.grads_2[param_name] is not None:
                    grad_3 = self.grads_3[param_name] if (self.grads_3 is not None and self.grads_3[param_name] is not None) else None
                    new_grad = self._rewrite_grad(self.grads_1[param_name], self.grads_2[param_name], grad_3)
                    param.grad.data.copy_(new_grad.data)
                    
    def reduce_bias(
        self,
        model_params_map,
        grads_1,
        grads_2,
        grads_3=None,
        param_partition=None,
    ):
        """
        Main method to perform bias reduction through selective parameter updating.
        Orchestrates the entire process of parameter selection, gradient rewriting, and optimization.

        Args:
        ---------
            model_params_map: Dictionary mapping parameter names to parameter tensors
            grads_1: Gradients from the first objective (e.g., male variant)
            grads_2: Gradients from the second objective (e.g., female variant)
            grads_3: Gradients from the third objective (e.g., neutral variant)
            param_partition: Optional parameter partitioning scheme
        """
        # Store gradients and partition information
        self.param_partition = param_partition
        self.grads_1 = grads_1
        self.grads_2 = grads_2
        self.grads_3 = grads_3
        self.params_to_keep = None

        # If k is specified, perform selective parameter updating
        if self.k is not None:
            # Select top k parameters based on gradient similarity
            self._top_k_params()
            self.update_model_param_grads(
                model_params_map=model_params_map,
            )
        # Otherwise, update all parameters
        else:
            self.update_model_param_grads(
                model_params_map=model_params_map,
            )

        # Apply the gradient updates to the model parameters
        self.optimizer.step()
