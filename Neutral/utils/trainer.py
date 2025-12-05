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

        for epoch in (range(self.start_at_epoch, self.num_epochs)):
            #logger.info(f'On epoch {epoch+1}/{self.num_epochs}')
            self.model.train()

            # Initialize batch counters and gradient accumulators
            curr_sim_batch_count = 0
            curr_male_grads = None
            curr_female_grads = None
            curr_neutral_grads = None

            for batch in self.dataloader:
                # Unpack batch data for all three gender variants
                # Each batch contains triples of sequences for multi-class bias comparison
                (
                (male_seqs, female_seqs, neutral_seqs),
                (male_att_mask, female_att_mask, neutral_att_mask),
                inds,
                (male_target, female_target, neutral_target),
                (male_labels, female_labels, neutral_labels)
                ) = batch

                self.optimizer.zero_grad()

                # Process male variant sequences
                if self.is_mlm: # masked
                    male_logits = self._mlm_backprop(
                                        input_ids=male_seqs,
                                        attention_mask=male_att_mask,
                                        indices=inds,
                                        target_tokens=male_target,
                                        vocab_size=vocab_size,
                                        do_backprop=not self.do_dynamic_gradient_selection)[1]

                # Capture gradients for male variant (static gradient selection)
                if not self.do_dynamic_gradient_selection:
                    male_grads = get_all_model_grads(self.model)

                # Process female variant sequences
                if self.is_mlm:
                    female_logits = self._mlm_backprop(
                                        input_ids=female_seqs,
                                        attention_mask=female_att_mask,
                                        indices=inds,
                                        target_tokens=female_target,
                                        vocab_size=vocab_size,)

                # Capture gradients for female variant (static gradient selection)
                if not self.do_dynamic_gradient_selection:
                    female_grads = get_all_model_grads(self.model)

                # Process neutral variant sequences
                if self.is_mlm:
                    neutral_logits = self._mlm_backprop(
                                        input_ids=neutral_seqs,
                                        attention_mask=neutral_att_mask,
                                        indices=inds,
                                        target_tokens=neutral_target,
                                        vocab_size=vocab_size,
                                        do_backprop=not self.do_dynamic_gradient_selection,)[1]

                # Capture gradients for neutral variant (static gradient selection)
                if not self.do_dynamic_gradient_selection:
                    neutral_grads = get_all_model_grads(self.model)

                # Dynamic gradient selection: minimize variance across all three gender variants
                if self.do_dynamic_gradient_selection:
                    # Compute mean logit across all three variants
                    mean_logits = (male_logits + female_logits + neutral_logits) / 3

                    # Create multipliers that push each variant toward the mean
                    # Positive multiplier if below mean (push up), negative if above mean (push down)
                    male_multiplier = mean_logits - male_logits
                    female_multiplier = mean_logits - female_logits
                    neutral_multiplier = mean_logits - neutral_logits

                    # Recompute with weighted gradients for male variant
                    self._mlm_backprop(
                        input_ids=male_seqs,
                        attention_mask=male_att_mask,
                        indices=inds,
                        target_tokens=male_target,
                        vocab_size=vocab_size,
                        do_backprop=True,
                        multiplier=male_multiplier)

                    male_grads = get_all_model_grads(self.model)

                    # Recompute with weighted gradients for female variant
                    self._mlm_backprop(
                        input_ids=female_seqs,
                        attention_mask=female_att_mask,
                        indices=inds,
                        target_tokens=female_target,
                        vocab_size=vocab_size,
                        do_backprop=True,
                        multiplier=female_multiplier)

                    female_grads = get_all_model_grads(self.model)

                    # Recompute with weighted gradients for neutral variant
                    self._mlm_backprop(
                        input_ids=neutral_seqs,
                        attention_mask=neutral_att_mask,
                        indices=inds,
                        target_tokens=neutral_target,
                        vocab_size=vocab_size,
                        do_backprop=True,
                        multiplier=neutral_multiplier)

                    neutral_grads = get_all_model_grads(self.model)

                # Accumulate gradients across batches
                curr_male_grads = accumulate_grad(curr_male_grads, male_grads)
                curr_female_grads = accumulate_grad(curr_female_grads, female_grads)
                curr_neutral_grads = accumulate_grad(curr_neutral_grads, neutral_grads)

                curr_sim_batch_count += self.batch_size

                # Apply optimizer step when similarity batch size is reached
                if self.sim_batch_size is not None and curr_sim_batch_count >= self.sim_batch_size:
                    self.reduce_bias(
                        model_params_map=params_map,
                        grads_1=curr_male_grads,
                        grads_2=curr_female_grads,
                        grads_3=curr_neutral_grads,
                        param_partition=param_partition)

                    curr_sim_batch_count = 0
                    curr_male_grads = None
                    curr_female_grads = None
                    curr_neutral_grads = None

            # Apply final optimizer step for the epoch if using accumulated gradients
            if self.sim_batch_size is None:
                self.reduce_bias(
                        model_params_map=params_map,
                        grads_1=curr_male_grads,
                        grads_2=curr_female_grads,
                        grads_3=curr_neutral_grads,
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

    
    