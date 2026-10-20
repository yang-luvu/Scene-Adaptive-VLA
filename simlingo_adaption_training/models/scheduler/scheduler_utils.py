import torch


SMART_BUDGET_NUM_ACTIONS = 10


def resolve_smart_budget_min(num_prefix_layers: int, num_hidden_layers: int) -> float:
    num_prefix = int(num_prefix_layers)
    num_hidden = int(num_hidden_layers)
    if not (2 <= num_prefix < num_hidden):
        raise ValueError(
            f"smart budget min requires 2 <= num_prefix_layers < num_hidden_layers, got "
            f"{num_prefix} and {num_hidden}"
        )
    if num_prefix % 2 != 0:
        raise ValueError(f"smart budget min requires even num_prefix_layers for half-split forward, got {num_prefix}")
    # Budget lower bound should never be below 0.2 and also respect prefix/total ratio.
    return float(max(0.2, float(num_prefix) / float(num_hidden)))


def build_smart_budget_action_table(
    budget_min: float,
    device: torch.device,
    dtype: torch.dtype,
    num_actions: int = SMART_BUDGET_NUM_ACTIONS,
) -> torch.Tensor:
    if not (0.0 <= float(budget_min) <= 1.0):
        raise ValueError(f"budget_min must be in [0, 1], got {budget_min}")
    n_actions = int(num_actions)
    if n_actions < 2:
        raise ValueError(f"num_actions must be >= 2, got {n_actions}")
    # Discrete action semantics: evenly cover the valid budget interval, including both endpoints.
    return torch.linspace(float(budget_min), 1.0, steps=n_actions, device=device, dtype=dtype)

        
def masked_gumbel_softmax_topk(logits, masks, k = 8, tau = 1, hard = False, eps = 1e-10, dim = -1, training = True):
    if training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
        logits = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

    logits = logits.masked_fill(masks.bool(), float('-inf')) # mask out the already picked items
    y_soft = logits.softmax(dim)

    y_hard = y_soft.masked_fill(masks.bool(), -1.0) # mask out the already picked items
    dim_size = int(logits.size(dim))
    k_eff = int(max(0, min(int(k), dim_size)))
    if k_eff == 0:
        return torch.zeros_like(logits)
    index = y_hard.topk(dim=dim, k=k_eff, largest=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)

    if hard:
        if training:
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_hard
    else:
        return y_soft

def posemb_sincos_1d(token_num, dim=256, temperature=10000, dtype=torch.float32):
    n = token_num

    n = torch.arange(n)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1. / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe
    
def budget_quantizing(budget, num_prefix_units=16, num_total_units=32):
    if not torch.all((budget >= 0) & (budget <= 1)):
        raise ValueError("Budget must be between 0 and 1.")
    units = torch.floor(num_total_units * budget) - num_prefix_units
    units = torch.relu(units)
    return units, units / (num_total_units - num_prefix_units + 1)
