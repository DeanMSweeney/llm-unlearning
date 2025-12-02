import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Literal, Union, Tuple, Dict, List

class PCGUMixin:
    """
    def __init__(
        self, 
        param_partition, 
        grads_1, 
        grads_2, 
        device,
        k: int=10000, 
        which_grad: str="negative",
    ):
        
        self.param_partition = param_partition
        self.self.grads_1 = self.grads_1
        self.self.grads_2 = self.grads_2
        self.k = k
        self.device = device
        self.which_grad = which_grad
        self.params_to_keep = None 
    """

    def top_k_params(self) -> List[float]: 
        """
        Args 
        ---------
        param_partition: (param_name, index) pairs denoting the partition of the weights
        grad_1: dict from param_name to gradient tensor
        grad_2: dict from param_name to gradient tensor
        """

        # Compute similiarity score 
        scores = []
        for param_name, indices in param_partition: 
            self.grads_1 = grad_1[param_name]
            self.grads_2 = grad_2[param_name]

            if self.grads_1 is None or self.grads_2 is None: # if gradients DNE, append large similarity score 
                scores.append(torch.Tensor([5]).squeeze()) 
                continue

            # index into the grads if this param is partitioned
            if indices is not None: 
                grad_1 = grad_1[indices]
                grad_2 = grad_2[indices]

            # compute and append cosine similarity score 
            cosine_sim = F.cosine_similarity(grad_1, grad_2, dim=-1).detach().cpu()
            scores.append(cosine_sim)
        
        # output top k most inlfuential parameters based on similiarities
        sim_stack = torch.stack(scores)
        param_partition = torch.Tensor(param_partition)

        # return k smallest elements (values, indices) in a tensor and filter params accordingly
        k_value, k_ind = sim_stack.topk(k, largest=False, sorted=True)
        self.params_to_keep = [param_partition[k_ind].detach().cpu()]
        
        return self.params_to_keep

    def _rewrite_grad(self, pos_grad, neg__grad):
        
        if self.which_grad == "negative":
            return self.grads_2
        if self.which_grad == "positve":
            return -self.grads_1
        elif self.which_grad == "neurtal":
            return -(self.grads_1+self.grads_2)
        else:
            raise AttributeError("Gradient method not selected")
        
    def update_model_param_grads(
        self,
        optimizer, 
        model_params_map, 
        new_grad_calc,
    ): 
    
        optimizer.zero_grad() # so that any grad not for param in params_to_keep is zero
        if self.params_to_keep is not None: 
            for param_name, indices in self.params_to_keep: 
                param = model_params_map[param_name]
                if indices is None: 
                    new_grad = self._rewrite_grad(pos_grad=self.grads_1[param_name], neg__grad=self.grads_2[param_name])
                    param.grad.data.copy_(new_grad.data)
                else: 
                    new_grad = self._rewrite_grad(pos_grad=self.grads_1[param_name], neg__grad=self.grads_2[param_name])
                    param.grad[indices] = new_grad.to(self.device)
        else: 
            for param_name, param in model_params_map.items(): 
                if self.grads_1[param_name] is not None and self.grads_2[param_name] is not None: 
                    new_grad = new_grad_calc(self.grads_1[param_name], self.grads_2[param_name])
                    param.grad.data.copy_(new_grad.data)

    def reduce_bias(
        self, 
        optimizer, 
        model_params_map, 
    ): 
        
        if self.k is not None: # do param partitioning    
            self.update_model_param_grads(
                optimizer=optimizer, 
                model_params_map=model_params_map, 
            )
        else: 
            self.update_model_param_grads(
                optimizer=optimizer, 
                model_params_map=model_params_map, 
            )

        optimizer.step()
