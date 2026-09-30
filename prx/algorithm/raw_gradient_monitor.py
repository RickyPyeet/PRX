from __future__ import annotations

import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

class RawGradientMonitor(Algorithm):
    """
    Algorithm to monitor both Transformer block gradients every `log_interval` number of optimizer steps and
    the global model's gradient.
    NOTE: to correctly inspect the Raw gradients the algorithm must be called BEFORE `Gradient Clipping`. So it must follow a FILO setup.
    """
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.param_mapping = {}
        self.n_blocks = None
        self.shard_group = None

        if self.log_interval <= 0:
            raise ValueError(f"Logging interval must be greater than 0, got {self.log_interval}")

    def match(self, event: Event, state: State):
        # Run between unscale_gradients and optimizer.step
        if event == Event.AFTER_TRAIN_BATCH:
            return True

        # Run before training for initial setup
        if event == Event.FIT_START:
            return True

        return False

    def apply(self, event: Event, state: State, logger: Logger):
        if event == Event.FIT_START:
            # Denoiser for the global l2_norm
            denoiser = state.model.denoiser
            # Transformer blocks for pin point l2_norm
            transformer_blocks = state.model.denoiser.blocks
            # FSDP2 mesh design for parallelism
            mesh_2d = state.device_mesh
            # N_blocks
            self.n_blocks = len(denoiser.blocks)

            # Create parameter-block mapping
            for block_idx, block in enumerate(transformer_blocks):
                for param in block.parameters():
                    self.param_mapping[id(param)] = block_idx
            
            # Extract mesh shard group
            self.shard_group = mesh_2d.get_group(mesh_dim = "data_parallel_shard")

        if event == Event.AFTER_TRAIN_BATCH:
            # Check if logging happens
            current_step = state.timestamp.batch.value + 1
            if current_step % self.log_interval != 0:
                return

            denoiser = state.model.denoiser
            accumulator = torch.zeros(
                self.n_blocks+1, # +1 for the global gradient 
                device = next(denoiser.parameters()).device,
                dtype = torch.float32
                )
            for param in denoiser.parameters():
                if param.requires_grad and param.grad is not None:
                    # Extract local grad and turn it into float32 for higher accuracy 
                    local_grad = param.grad.to_local().float()
                    # Compute squared sum for future l2_norm
                    squared_sum = local_grad.square().sum()
                    # All denoiser parameters contribute to the global l2_norm
                    accumulator[-1] += squared_sum
                    # Transformer parameters contribute to their corresponding l2_norms
                    block_idx = self.param_mapping.get(id(param))

                    if block_idx is not None:
                        accumulator[block_idx] += squared_sum

            # All reduce the sums of current GPU's shards over mesh_2d shard_group
            torch.distributed.all_reduce(
                accumulator,
                op = torch.distributed.ReduceOp.SUM,
                group = self.shard_group
            )

            # Compute final l2_norm
            accumulator = accumulator.sqrt()

            # Log metrics
            metrics = {}
            for block_idx in range(self.n_blocks):
                metrics[f'grad_norm/block_{block_idx}'] = accumulator[block_idx]
            metrics['grad_norm/global'] = accumulator[-1]

            logger.log_metrics(metrics)



