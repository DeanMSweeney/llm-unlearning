from typing import Optional, Literal, Union, Tuple, Dict, List
import logging
from tqdm import tqdm


from utils.mixins import PCGUMixin
from utils.utils import save_model, create_param_partition, get_params_map, get_all_model_grads, accumulate_grad

logger = logging.getLogger(__name__)

class Unbias(PCGUMixin):
    """
    Trainer class for model retraining with optional dynamic gradient selection.
    Inherits from PCGUMixin for parameter selection capabilities.

    Args:
    -----------
        model: The language model to train
        tokenizer: Tokenizer for the model
        optimizer: Optimization algorithm
        dataloader: DataLoader for training data
        batch_size: Number of samples per batch
        device: Device to run training on (CPU/GPU)
        do_dynamic_gradient_selection: Whether to dynamically select parameters based on gradients
        dedupe: Deduplication strategy identifier
        model_name: Name/identifier of the model being trained
        agg_dim: Dimension to aggregate gradients along
        start_at_epoch: Epoch to start training from
        sim_batch_size: Batch size for similarity computations (-1 for default)
        proportion_dev: Proportion of data to use for development/validation
        k: Number of top parameters to keep for gradient selection
        which_grad: Type of gradients to use ('negative' for unlearning)
    
    """
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        dataloader,
        batch_size: int,
        num_epochs, 
        device,
        is_mlm: bool=True,
        do_dynamic_gradient_selection: bool=False,
        dedupe='',
        model_name='',
        agg_dim: int=-1,
        start_at_epoch: int=0,
        sim_batch_size: int=-1,
        proportion_dev=0.5,
        k: int=10000,
        which_grad: str="advantaged",
    ):
      
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.is_mlm = is_mlm
        self.do_dynamic_gradient_selection = do_dynamic_gradient_selection
        self.dedupe = dedupe
        self.model_name = model_name
        self.agg_dim = agg_dim
        self.start_at_epoch = start_at_epoch
        self.sim_batch_size = sim_batch_size
        self.proportion_dev = proportion_dev
        self.k = k
        self.which_grad = which_grad
        self.params_to_keep = None 
    
        
    def retrain(self):
        logger.info('retraining mlm')
        
        if self.sim_batch_size == -1: 
            self.sim_batch_size =self.batch_size

        if self.do_dynamic_gradient_selection: 
            self.which_grad = "combined"

        if self.sim_batch_size is not None and (self.sim_batch_size<=0 or self.sim_batch_size%self.batch_size!=0): 
            raise ValueError(f'Batch size for computing similarity is invalid: {self.sim_batch_size}')

        vocab_size = len(self.tokenizer)
        params_map = get_params_map(self.model)
        param_partition = create_param_partition(params_map, dim_to_agg=self.agg_dim)

        for epoch in tqdm(range(self.start_at_epoch, self.num_epochs)):
            #logger.info(f'On epoch {epoch+1}/{self.num_epochs}')
            self.model.train()

            # Initialize batch counters and gradient accumulators
            curr_sim_batch_count = 0
            curr_disadv_grads = None
            curr_adv_grads = None

            for batch in self.dataloader:
                # Unpack batch data for both disadvantaged and advantaged groups
                # Each batch contains paired sequences for bias comparison
                (
                (disadv_seqs, adv_seqs),
                (disadv_att_mask, adv_att_mask),
                inds,
                (disadv_target, adv_target),
                (disadv_labels, adv_labels)
                ) = batch

                self.optimizer.zero_grad()

                # Process disadvantaged group sequences
                if self.is_mlm: # masked 
                    disadv_logits = self._mlm_backprop(
                                        input_ids=disadv_seqs,
                                        attention_mask=disadv_att_mask,
                                        indices=inds,
                                        target_tokens=disadv_target,
                                        vocab_size=vocab_size,
                                        do_backprop=not self.do_dynamic_gradient_selection)[1]
                else: # causal
                    self._lm_backprop(
                                        input_ids=disadv_seqs,
                                        attention_mask=disadv_att_mask,
                                        labels=disadv_labels,)

                # Capture gradients for disadvantaged group (static gradient selection)
                if not self.do_dynamic_gradient_selection:
                    disadv_grads = get_all_model_grads(self.model)

                # Process advantaged group sequences
                if self.is_mlm:
                    adv_logits = self._mlm_backprop(
                                        input_ids=adv_seqs,
                                        attention_mask=adv_att_mask,
                                        indices=inds,
                                        target_tokens=adv_target,
                                        vocab_size=vocab_size,
                                        do_backprop=not self.do_dynamic_gradient_selection,)[1]
                else:
                    self._lm_backprop(
                        input_ids=adv_seqs,
                        attention_mask=adv_att_mask,
                        labels=adv_labels,)

                # Capture gradients for advantaged group (static gradient selection)
                if not self.do_dynamic_gradient_selection:
                    adv_grads = get_all_model_grads(self.model)

                # Dynamic gradient selection: only update parameters where disadvantaged < advantaged
                if self.do_dynamic_gradient_selection:
                    # Identify truly disadvantaged examples (lower logits than advantaged)
                    disadv_actually_disadv = disadv_logits < adv_logits
                    # Create multiplier: +1 for truly disadvantaged, -1 otherwise
                    multiplier = disadv_actually_disadv.float() * 2 - 1

                    # Recompute with weighted gradients for disadvantaged group
                    self._mlm_backprop(
                        input_ids=disadv_seqs,
                        attention_mask=disadv_att_mask,
                        indices=inds,
                        target_tokens=disadv_target,
                        vocab_size=vocab_size,
                        do_backprop=True,
                        multiplier=multiplier) # add multiplier here 
                    
                    disadv_grads = get_all_model_grads(self.model)

                    # Recompute with inverted weighted gradients for advantaged group
                    self._mlm_backprop(
                        input_ids=adv_seqs,
                        attention_mask=adv_att_mask,
                        indices=inds,
                        target_tokens=adv_target,
                        vocab_size=vocab_size,
                        do_backprop=True,
                        multiplier=-multiplier)
                    
                    adv_grads = get_all_model_grads(self.model)

                # Accumulate gradients across batches
                curr_disadv_grads = accumulate_grad(curr_disadv_grads, disadv_grads)
                curr_adv_grads = accumulate_grad(curr_adv_grads, adv_grads)

                curr_sim_batch_count += self.batch_size

                # Apply optimizer step when similarity batch size is reached
                if self.sim_batch_size is not None and curr_sim_batch_count >= self.sim_batch_size:
                    self.reduce_bias(
                        model_params_map=params_map, 
                        grads_1=curr_disadv_grads, 
                        grads_2=curr_adv_grads, 
                        param_partition=param_partition)
                    
                    curr_sim_batch_count = 0
                    curr_disadv_grads = None
                    curr_adv_grads = None

            # Apply final optimizer step for the epoch if using accumulated gradients
            if self.sim_batch_size is None:
                self.reduce_bias(
                        model_params_map=params_map, 
                        grads_1=curr_disadv_grads, 
                        grads_2=curr_adv_grads, 
                        param_partition=param_partition)

            saved_model_dir = save_model(model=self.model, tokenizer=self.tokenizer, epoch=epoch+1, dedupe=self.dedupe)

    
    def _mlm_backprop(
        self,
        input_ids,
        attention_mask,
        indices,
        target_tokens,
        vocab_size,
        do_backprop=True,
        multiplier=None
    ):
        """
        Perform backpropagation for masked language modeling (MLM).

        Args:
        ----------
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            indices: Positions of masked tokens in the sequence
            target_tokens: Target token IDs for the masked positions
            vocab_size: Size of the model's vocabulary
            do_backprop: Whether to perform backward pass
            multiplier: Optional multiplier to scale logits (for weighted unlearning)

        Returns:
            Tuple of (final_output, logits) where final_output is the sum of target logits
        """
        # Move all tensors to the appropriate device
        input_ids = input_ids.to(self.device)
        indices = indices.to(self.device)
        target_tokens = target_tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)

        self.model.zero_grad()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract logits based on model architecture (BERT vs RoBERTa)
        if 'roberta' not in self.model_name:
            logits = outputs.prediction_logits
        else:
            logits = outputs.logits

        # Gather logits at masked positions and for target tokens
        indices = indices.unsqueeze(-1).repeat(1, vocab_size).unsqueeze(1)
        target_tokens = target_tokens.unsqueeze(-1)
        logits = logits.gather(1, indices)

        # Handle shape for single-item batches
        unsqueeze_later = logits.shape[0]==1
        logits = logits.squeeze()
        if unsqueeze_later:
            logits = logits.unsqueeze(0)
        logits = logits.gather(1, target_tokens)

        # Apply optional scaling multiplier
        if multiplier is not None:
            logits = logits*multiplier

        final_output = logits.sum()

        if do_backprop:
            final_output.backward()

        return final_output, logits

    def _lm_backprop(
        self,
        input_ids,
        attention_mask,
        labels,
    ):
        """
        Perform backpropagation for causal language modeling.

        Args:
        ----------
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            labels: Target labels for language modeling loss

        Returns:
            Negative loss value (used for gradient ascent in unlearning)
        """
        # Move all tensors to the appropriate device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        self.model.zero_grad()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Negate loss for gradient ascent (unlearning)
        loss = -outputs.loss
        loss.backward()

        return loss
    
    