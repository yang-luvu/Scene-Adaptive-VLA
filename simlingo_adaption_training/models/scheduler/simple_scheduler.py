import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .scheduler_utils import *


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class SimpleScheduler_L(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
        num_sub_layer = self.num_hidden_layers - self.num_prefix_layers


        self.mlp_head = nn.Linear(self.hidden_size, num_sub_layer, bias=self.bias)
        self.scheduler_up_proj = FeedForward(256, self.hidden_size, self.hidden_size)
        
        self.init_weights()

    def set_tau(self, tau):
        self.tau = tau

    def budget_encoding(self, budget):
        quantized_budget = budget_quantizing(budget, self.num_prefix_layers, self.num_hidden_layers)[1]

        # Scale the batch budget to a range of [0, 2π]
        scaled_values = quantized_budget * 2 * torch.pi  # Shape: [batch_size]

        # Generate frequency indices to create a diverse range of sine and cosine values
        frequencies = 1 / (10000 ** (torch.arange(128).to(quantized_budget.device) / 128))  # 128 frequencies for each sin and cos

        # Expand dimensions to compute sine and cosine for each value in the batch with each frequency
        # `scaled_values[:, None]` adds a dimension to match (batch_size, 1) with (128), resulting in (batch_size, 128)
        sin_values = torch.sin(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]
        cos_values = torch.cos(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]

        # Concatenate sin and cos values along the last dimension to get a tensor of size [batch_size, 256]
        budget_emb = torch.cat((sin_values, cos_values), dim=1).to(quantized_budget.dtype)  # Shape: [batch_size, 256]
        
        budget_emb = self.scheduler_up_proj(budget_emb)
        return budget_emb

    def forward(self, x, budget):
        '''
         x: budget token [bs, hidden_size]
         budget: normalized budget
        
         return: execution plan, size [batch_size, num_hidden_layers, 2, num_attention_heads]
        '''
        budget = budget_quantizing(budget, self.num_prefix_layers, self.num_hidden_layers)[0]
        logits = self.mlp_head(x)
        output_samples = []
        for logits_, budget_ in zip (logits, budget):
            sample = masked_gumbel_softmax_topk(
                logits=logits_,
                masks=torch.zeros_like(logits_),
                k=int(budget_.item()),
                tau=self.tau,
                hard=self.is_hard,
                training=self.training,
            )
            output_samples.append(sample)
        
        output_samples = torch.stack(output_samples)
        output_samples = output_samples[:,:,None,None].repeat(1, 1, 2, self.num_attention_heads)

        prefix_execution_plan = self.get_prefix_execution_plan(output_samples)

        output_samples = torch.cat([prefix_execution_plan, output_samples], 1)
        return output_samples
    
    def get_prefix_execution_plan(self, output_samples):
        new_shape = list(output_samples.shape)
        new_shape[1] = self.num_prefix_layers

        prefix_execution_plan = torch.ones(new_shape, device=output_samples.device, dtype=output_samples.dtype)
        return prefix_execution_plan

    def get_random_budget(self, batch_size):
        budget = torch.randint(self.num_prefix_layers, self.num_hidden_layers + 1, (batch_size,)) / self.num_hidden_layers
        return budget

    def init_weights(self):
            """Initialize linear weights normally and zero their biases."""
            nn.init.normal_(self.mlp_head.weight, mean=0.0, std=0.02)
            if self.mlp_head.bias is not None:
                nn.init.zeros_(self.mlp_head.bias)

            for m in self.scheduler_up_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.zeros_(m.bias)
                    nn.init.ones_(m.weight)
    

class SimpleScheduler_H(nn.Module):
    def __init__(self, config, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.num_prefix_layers = config.num_prefix_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.num_sub_layer = config.num_hidden_layers - config.num_prefix_layers
        self.num_attention_heads = config.num_attention_heads
        self.is_hard = is_hard
        self.tau = tau
        self.rank = config.scheduler_rank

        self.mlp_head = nn.Linear(config.hidden_size, self.num_sub_layer * self.num_attention_heads * 2, bias=bias)
        self.scheduler_up_proj = FeedForward(256, config.hidden_size, config.hidden_size)

    def set_tau(self, tau):
        self.tau = tau

    def budget_encoding(self, budget):
        quantized_budget = budget_quantizing(
            budget,
            self.num_prefix_layers * self.num_attention_heads // self.rank,
            self.num_hidden_layers * self.num_attention_heads // self.rank,
        )[1]

        # Scale the batch budget to a range of [0, 2π]
        scaled_values = quantized_budget * 2 * torch.pi  # Shape: [batch_size]

        # Generate frequency indices to create a diverse range of sine and cosine values
        frequencies = 1 / (10000 ** (torch.arange(128).to(quantized_budget.device) / 128))  # 128 frequencies for each sin and cos

        # Expand dimensions to compute sine and cosine for each value in the batch with each frequency
        # `scaled_values[:, None]` adds a dimension to match (batch_size, 1) with (128), resulting in (batch_size, 128)
        sin_values = torch.sin(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]
        cos_values = torch.cos(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]

        # Concatenate sin and cos values along the last dimension to get a tensor of size [batch_size, 256]
        budget_emb = torch.cat((sin_values, cos_values), dim=1).to(quantized_budget.dtype)  # Shape: [batch_size, 256]
        
        budget_emb = self.scheduler_up_proj(budget_emb)
        return budget_emb
    
    def forward(self, x, budget):
        budget = budget_quantizing(
            budget,
            self.num_prefix_layers * self.num_attention_heads // self.rank,
            self.num_hidden_layers * self.num_attention_heads // self.rank,
        )[0]
        logits = self.mlp_head(x)
        output_samples = []
        for logits_, budget_ in zip (logits, budget):
            logits_ = logits_.view(self.num_sub_layer, self.num_attention_heads * 2)
            sample = masked_gumbel_softmax_topk(
                logits=logits_.view(-1),
                masks=torch.zeros_like(logits_).view(-1),
                k=int(budget_.item()) * int(self.rank) * 2,
                tau=self.tau,
                hard=self.is_hard,
                training=self.training,
            )
            output_samples.append(sample)
        
        output_samples = torch.stack(output_samples)
        output_samples = output_samples.view(output_samples.size(0), self.num_sub_layer, 2, -1)

        prefix_execution_plan = self.get_prefix_execution_plan(output_samples)

        output_samples = torch.cat([prefix_execution_plan, output_samples], 1)
        return output_samples
    
    def get_prefix_execution_plan(self, output_samples):
        new_shape = list(output_samples.shape)
        new_shape[1] = self.num_prefix_layers

        prefix_execution_plan = torch.ones(new_shape, device=output_samples.device, dtype=output_samples.dtype)
        return prefix_execution_plan

    def get_random_budget(self, batch_size):
        budget = torch.randint(
            self.num_prefix_layers * self.num_attention_heads // self.rank,
            self.num_hidden_layers * self.num_attention_heads // self.rank + 1,
            (batch_size,),
        ) / (self.num_hidden_layers * self.num_attention_heads // self.rank)
        return budget
