from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from simlingo_adaption_training.utils.custom_types import DrivingExample

def cross_track_error(points: Tensor, path: Tensor):
    """
    Computes the cross track error between a set of points and a path.

    Args:
        points: The set of points to compute the cross track error for with shape [b, n, 2].
        path: The path to compute the cross track error with with shape [b, m, 2]. The path
            can contain nan values which indicates that the path is not available for that position.

    Returns:
        The cross track error for each point in the set of points with shape [b, n].
    """

    points, path = points.float(), path.float()

    ind = torch.arange(path.size(0), device=path.device)[:, None]
    closest = torch.cdist(points, path).nan_to_num_(torch.inf).argmin(-1)
    pt0 = path[ind, (closest - 1).clamp_min(0)]
    pt1 = path[ind, closest]
    pt2 = path[ind, (closest + 1).clamp_max(path.size(1) - 1)]

    tangent = (pt2 - pt1).nan_to_num_(0.0) + (pt1 - pt0).nan_to_num_(0.0)
    normal = torch.stack((tangent[..., 1], -tangent[..., 0]), dim=-1)
    normal = normal / normal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-2)

    return (points - pt1).mul(normal).sum(-1).abs()

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
    
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WaypointInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, N, 2] and returns an output of shape [B, N, token_size]
    Args:
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
        norm_layer: the `Module` to use to normalize the values of the input tensor.
    """
    
    def __init__(
        self, token_size: int = 258, hidden_size: int = 64, hidden_size2: int = 128, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer

        self.mlp = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, hidden_size2), nn.ReLU(True), nn.Linear(hidden_size2, token_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input with dims [B, N, 2]

        Returns:
            Output with dims [B, N, token_size]
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x)
        return x


class DrivingAdaptor(nn.Module):
    def __init__(self, 
                hidden_size: int, 
                mlp_dim=256, 
                predict_route_as_wps=False, 
                speed_wps_mode=False,
            ):
        super().__init__()
        self.heads = {}
        self.order = []

        self.speed_wps_mode = speed_wps_mode
        self.predict_route_as_wps = predict_route_as_wps

        if predict_route_as_wps:
            self.future_waypoints = 20
            self.query_embeds_wps = nn.Parameter(0.02 * torch.randn((1, self.future_waypoints, hidden_size)))
            self.route_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim*2), nn.SiLU(True),nn.Linear(mlp_dim*2, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, 2, bias=False)
            )
            
            self.queries = {'route': self.query_embeds_wps} # route predict token, learnable token
            self.sizes = {'route': self.future_waypoints}
            self.heads["route"] = self.route_head
            self.order.append('route')

        if speed_wps_mode == '2d':
            dim = 2
        elif speed_wps_mode == '1d':
            dim = 1
        else:
            raise ValueError(f"speed_wps_mode must be '1d' or '2d', not {speed_wps_mode}")
        self.future_speed_waypoints = 10 #TODO: read from config
        self.query_embeds_speed = nn.Parameter(0.02 * torch.randn((1, self.future_speed_waypoints, hidden_size))) # speed predict token, learnable token
        self.speed_wps_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, dim, bias=False)
            )
        self.heads["speed_wps"] = self.speed_wps_head
        self.queries['speed_wps'] = self.query_embeds_speed
        self.sizes['speed_wps'] = self.future_speed_waypoints
        self.order.append('speed_wps')


    def forward(self, 
            driving_example: DrivingExample,
            **kwargs
            ) -> Dict[str, Tensor]:

        try:
            driving_input = driving_example.driving_input
        except AttributeError:
            driving_input = driving_example
        
        b = driving_input.camera_images.shape[0] # plt / wandb
        inputs = None

        for input_type in self.order:
            query_embed = self.queries[input_type]
            if inputs is None:
                inputs = query_embed.expand(b, -1, -1)
            else:
                inputs = torch.cat((inputs, query_embed.expand(b, -1, -1)), dim=1)

        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool)

        return {"inputs": inputs, "inputs_mask": inputs_mask}

    def get_predictions(
        self, 
        features: Tensor,
        logits: Optional[Tensor] = None
    ) -> Dict:

        current_index = 0
        predictions = {}
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]

            feature = features[:, current_index: current_index + size]
            prediction = self.heads[input_type](feature).cumsum(1)

            predictions[input_type] = prediction
            current_index += size
        
        return predictions


    def compute_loss(
        self, adaptor_features: Tensor, adaptor_logits: Tensor, _inputs: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        label = example.driving_label
        assert label is not None
        
        if self.predict_route_as_wps:
            label_route = label.path
        else:
            label_route = None

        if self.speed_wps_mode == '2d':
            label_speed_wps = label.waypoints[:, : self.future_waypoints + 1]
        elif self.speed_wps_mode == '1d':
            label_speed_wps = label.waypoints_1d
        else:
            label_speed_wps = None

        current_index = 0
        loss_dict = {}
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]
            features_tmp = adaptor_features[:, current_index: current_index + size]
            label = locals()[f'label_{input_type}']

            prediction = self.heads[input_type](features_tmp).cumsum(1)
            loss = F.smooth_l1_loss(prediction, label, reduction="none").sum(-1)
            
            loss_dict[f"{input_type}_loss"] = (loss, torch.ones_like(loss, dtype=torch.long))
            loss_dict[f"{input_type}_prediction"] = prediction
            loss_dict[f"{input_type}_label"] = label
            current_index += size

        return loss_dict


class LanguageAdaptor(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.embed_tokens = language_model.model.embed_tokens
        if hasattr(language_model.model, "lm_head"):
            self.lm_head = language_model.model.lm_head
        elif hasattr(language_model.model, "embed_out"):
            self.lm_head = language_model.model.embed_out
        elif hasattr(language_model.model.base_model.model, 'output'):
            self.lm_head = language_model.model.base_model.model.output
        else:
            raise ValueError("Language model must have `lm_head` or `embed_out` attribute.")


    def forward(self, example: DrivingExample, inference=False, **kwargs) -> Dict[str, Tensor]:
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
            
        b = driving_input.camera_images.size(0)
        
        if inference:
            label = driving_input.prompt_inference
        else:
            label = driving_input.prompt
        
        if label is not None:
            ids = label.phrase_ids.long()
            ids_valid = label.phrase_valid
            ids_mask = label.loss_masking

        inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))
        return {"inputs": inputs, "inputs_mask": ids_valid, "_ids": ids, "_ids_mask": ids_mask}

    def compute_loss(
        self, adaptor_features: Tensor, adaptor_logits: Tensor, inputs: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        del example

        if adaptor_logits is None:
            adaptor_logits = self.lm_head(outputs[:, :-1])
        else:
            adaptor_logits = adaptor_logits[:, :-1]
        labels = torch.where(inputs["_ids_mask"], inputs["_ids"], -1)
        # Shift by 1 for next token prediction
        labels = labels[:, 1:]
        language_loss = F.cross_entropy(
            adaptor_logits.flatten(0, -2), labels.flatten(), ignore_index=-1, reduction="none"
        ).view_as(labels)
        return {"language_loss": (language_loss, labels.ne(-1))}


class BudgetAdaptor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, example: DrivingExample, budget_assigner=None, **kwargs) -> Dict[str, Tensor]:
        """Encode the current runtime budget as a conditioning token."""
        del kwargs

        if budget_assigner is None:
            raise ValueError("budget_assigner is required for BudgetAdaptor")

        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
        
        bs = driving_input.camera_images.shape[0]
        device = driving_input.camera_images.device

        budget_tensor = budget_assigner.get_runtime_budget_values(batch_size=bs)
        if budget_tensor is None:
            raise ValueError("runtime budget is missing for BudgetAdaptor")
        budget_tensor = budget_tensor.to(device=device)

        inputs = budget_assigner.encode_budget_token(budget_tensor)
        
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1) # [Batch, 1, Hidden]

        inputs_mask = torch.ones((bs, 1), dtype=torch.bool, device=device)

        return {
            "inputs": inputs, 
            "inputs_mask": inputs_mask,
            "values": budget_tensor
        }

    def compute_loss(self, *args, **kwargs):
        return {}

class SceneAwareAdaptor(nn.Module):
    """Reserve tokens populated later by the smart assigner."""
    def __init__(self, hidden_size: int, num_tokens: int = 3):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)

    def forward(self, example: DrivingExample, **kwargs) -> Dict[str, Tensor]:
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example

        bs = driving_input.camera_images.shape[0]
        device = driving_input.camera_images.device
        dtype = driving_input.camera_images.dtype
        inputs = torch.zeros((bs, self.num_tokens, self.hidden_size), dtype=dtype, device=device)
        inputs_mask = torch.ones((bs, self.num_tokens), dtype=torch.bool, device=device)
        return {"inputs": inputs, "inputs_mask": inputs_mask}

    def compute_loss(self, *args, **kwargs):
        return {}


class AssignQueryAdaptor(nn.Module):
    """Inject learnable budget and path queries."""
    def __init__(self, hidden_size: int, query_init_std: float = 0.02):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.query_budget = nn.Parameter(float(query_init_std) * torch.randn((1, 1, self.hidden_size)))
        self.query_path = nn.Parameter(float(query_init_std) * torch.randn((1, 1, self.hidden_size)))

    def forward(self, example: DrivingExample, **kwargs) -> Dict[str, Tensor]:
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example

        bs = driving_input.camera_images.shape[0]
        # shape: (B, 2, C)
        inputs = torch.cat(
            [
                self.query_budget.expand(bs, -1, -1),
                self.query_path.expand(bs, -1, -1),
            ],
            dim=1,
        )
        inputs_mask = torch.ones((bs, 2), dtype=torch.bool, device=inputs.device)
        return {"inputs": inputs, "inputs_mask": inputs_mask}

    def compute_loss(self, *args, **kwargs):
        return {}


class AdaptorList(nn.Module):
    """
    Each adaptor is responsible for converting a driving example
    to a sequence of tokens and computing the loss on the token outputs.
    Adaptors are only used during training.
    """

    def __init__(
        self,
        driving: Optional[DrivingAdaptor] = None,
        language: Optional[LanguageAdaptor] = None,
        sceneaware: Optional[SceneAwareAdaptor] = None,
        assignquery: Optional[AssignQueryAdaptor] = None,
        budget: Optional[BudgetAdaptor] = None,
    ):
        super().__init__()
        self.driving = driving
        self.language = language
        self.sceneaware = sceneaware
        self.assignquery = assignquery
        self.budget = budget
    @property
    def adaptors(self):
        dct: Dict[str, Adaptor] = {}
        if self.language is not None:
            dct["language"] = self.language
        if self.driving is not None:
            dct["driving"] = self.driving
        if self.sceneaware is not None:
            dct["sceneaware"] = self.sceneaware
        if self.assignquery is not None:
            dct["assignquery"] = self.assignquery
        if self.budget is not None:
            dct["budget"] = self.budget
        return dct

    def _resolve_active_adaptor_keys(self, active_adaptor_keys: Optional[List[str]] = None) -> List[str]:
        all_keys = list(self.adaptors.keys())
        if active_adaptor_keys is None:
            return all_keys
        active_keys = list(active_adaptor_keys)
        for key in active_keys:
            if key not in self.adaptors:
                raise ValueError(f"Unknown adaptor key: {key}")
        return active_keys

    def forward(
        self,
        example: DrivingExample,
        active_adaptor_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Construct input embeddings for the given driving example.
        """

        input_dict: Dict[str, Tensor] = {}
        inputs_list: List[Tensor] = []
        inputs_mask_list: List[Tensor] = []

        active_keys = self._resolve_active_adaptor_keys(active_adaptor_keys)

        for key in active_keys:
            adaptor = self.adaptors[key]
            adaptor_input_dict = adaptor.forward(example, **kwargs)
            inputs_list.append(adaptor_input_dict["inputs"])
            inputs_mask_list.append(adaptor_input_dict["inputs_mask"])
            input_dict.update({key + "_" + k: v for k, v in adaptor_input_dict.items()}) # all

        inputs = torch.cat(inputs_list, dim=1)
        inputs_mask = torch.cat(inputs_mask_list, dim=1)
        split_sizes = torch.as_tensor([x.size(1) for x in inputs_list])
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]

        # Apply random permutation of modalities during training
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1)
        # Apply permutation to move invalid tokens to end of sequence
        valid_perm = inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)
        perm = rand_perm.gather(1, valid_perm)

        input_dict["inputs"] = inputs[arange, perm]
        input_dict["inputs_mask"] = inputs_mask[arange, perm]
        input_dict["perm"] = perm
        input_dict["split_sizes"] = split_sizes
        input_dict["active_adaptor_keys"] = active_keys
        return input_dict

    def compute_loss(
        self, features: Tensor, logits: Tensor, input_dict: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Distributes the output embeddings from the transformer to
        the correct loss function and returns a dictionary of losses.
        """

        features_by_adaptor = self.split_outputs_by_adaptor(input_dict, features)
        logits_by_adaptor = self.split_outputs_by_adaptor(input_dict, logits)

        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}

        # Compute loss in each adaptor
        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}
        active_keys = input_dict.get("active_adaptor_keys", list(self.adaptors.keys()))
        for key in active_keys:
            adaptor = self.adaptors[key]
            adaptor_input_dict = _gather_from_dict(input_dict, key + "_")
            adaptor_features = features_by_adaptor[key]
            adaptor_logits = logits_by_adaptor[key]
            losses = adaptor.compute_loss(adaptor_features, adaptor_logits, adaptor_input_dict, example)
            loss_dict.update(losses)

        return loss_dict

    def split_outputs_by_adaptor(self, input_dict: Dict[str, Tensor], outputs: Tensor) -> Dict[str, Tensor]:
        """
        Splits the output tensor into the correct output for each adaptor, according to the
        split_sizes in the input_dict.
        """
        # First reverse permutation
        inv_perm = input_dict["perm"].argsort(-1)
        arange = torch.arange(inv_perm.size(0), device=inv_perm.device)[:, None]
        outputs = outputs[arange, inv_perm]

        # Now split output for each adaptor
        split_sizes = [int(x) for x in input_dict["split_sizes"]]
        outputs_list = list(outputs.split(split_sizes, dim=1))
        active_keys = input_dict.get("active_adaptor_keys", list(self.adaptors.keys()))
        return {key: outputs_list[i] for i, key in enumerate(active_keys)}


def _gather_from_dict(d: Dict[str, Tensor], prefix: str):
    out: Dict[str, Tensor] = {}  # dict comprehensions with if not supported
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out


def refresh_adaptor_inputs(adaptor_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Rebuild adaptor inputs while preserving the configured permutation."""
    active_keys = adaptor_dict.get("active_adaptor_keys")
    if active_keys is None:
        raise ValueError("active_adaptor_keys missing from adaptor_dict")

    inputs_list: List[Tensor] = []
    masks_list: List[Tensor] = []
    for key in active_keys:
        inputs_key = f"{key}_inputs"
        mask_key = f"{key}_inputs_mask"
        if inputs_key not in adaptor_dict or mask_key not in adaptor_dict:
            raise ValueError(f"Missing adaptor segment for key={key}: expected {inputs_key} and {mask_key}")
        inputs_list.append(adaptor_dict[inputs_key])
        masks_list.append(adaptor_dict[mask_key])

    inputs = torch.cat(inputs_list, dim=1)
    inputs_mask = torch.cat(masks_list, dim=1)
    arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]
    perm = adaptor_dict["perm"]
    adaptor_dict["inputs"] = inputs[arange, perm]
    adaptor_dict["inputs_mask"] = inputs_mask[arange, perm]
    return adaptor_dict


def replace_placeholder_tokens(
    adaptor_dict: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    placeholder_values: Optional[List[dict]] = None,
    wp_encoder: Optional[nn.Module] = None,
    image_encoder: Optional[nn.Module] = None,
):
    """Replace language placeholders and rebuild the permuted adaptor input."""
    if 'tokenizer' in image_encoder.processor.__dict__:
        image_encoder.tokenizer = image_encoder.processor.tokenizer
    else:
        image_encoder.tokenizer = image_encoder.processor

    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    img_context_token_id = image_encoder.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    image_encoder.img_context_token_id = img_context_token_id
    
    output_attentions = output_attentions if output_attentions is not None else image_encoder.model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else image_encoder.model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else image_encoder.model.config.use_return_dict

    if inputs_embeds is None:
        # 1. Extract the input embeddings
        # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
        inputs_embeds = adaptor_dict['language_inputs']
        input_ids = adaptor_dict['language__ids']
        inputs_embeds_dtype = inputs_embeds.dtype
        # `index_put` under bf16 autocast is fragile in backward. Do the writeback in fp32
        # and cast back to the original dtype after all placeholder/image replacements.
        inputs_embeds_work = inputs_embeds.to(dtype=torch.float32)
        
        # 2a replace placeholder
        smallest_added_id = image_encoder.tokenizer.additional_special_tokens_ids[0]
        special_ids = torch.tensor(list(set(input_ids[(input_ids >= smallest_added_id)].tolist())), device=input_ids.device)
        special_ids = special_ids.view(-1, 1, 1)
        batch_size, seq_len = input_ids.shape

        if special_ids.size(0) > 0 and len(placeholder_values) > 0:
            wp_encoder_dtype = wp_encoder.mlp[0].weight.dtype

            # Create a mask where the special_ids are located
            mask = input_ids == special_ids

            # Convert the mask to float and use torch.cumsum to get cumulative sum along the sequence length dimension
            cumsum_mask = torch.cumsum(mask.float(), dim=2)

            # Create a mask to get the first occurrence by checking where cumsum is 1
            first_occurrence_mask = (cumsum_mask == 1) & mask

            # Use torch.argmax to get the indices of the first occurrence
            first_occurrences = torch.argmax(first_occurrence_mask.float(), dim=2)
            # swap the dimensions to get the batch and sequence length
            first_occurrences = first_occurrences.transpose(0, 1)

            # get coords from label.placeholder_values with batch and special_id as key
            special_token_pos = first_occurrences.nonzero()

            coords = [torch.tensor(placeholder_values[b_id][special_ids[key_id].item()], device=input_ids.device, dtype=wp_encoder_dtype) for key_id, b_id in zip(special_token_pos[:, 1], special_token_pos[:, 0])]
            coords_length_org = [len(coord) for coord in coords]
            coords = torch.cat(coords)
            wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)
            wp_embeds = torch.split(wp_embeds, coords_length_org)

            first_occurrences_filtered = [first_occurrences[i] for i in special_token_pos[:, 0]]

            with torch.autocast(device_type=inputs_embeds_work.device.type, enabled=False):
                for i, (pos, first_occurrence) in enumerate(zip(special_token_pos, first_occurrences_filtered)):
                    start = first_occurrence[pos[1]]
                    end = start + coords_length_org[i]
                    inputs_embeds_work[pos[0], start:end] = wp_embeds[i].to(dtype=inputs_embeds_work.dtype)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
            all_pixel_values = [pixel_values]
                
            all_image_features = []
            all_feature_lens = []
            _, N_embed, C_embed = inputs_embeds_work.shape
            
            for pixel_values_tmp in all_pixel_values:
                BS, T, NP, C, H, W = pixel_values_tmp.shape
                assert T == 1, "Only one frame is supported for now"
                # for multi-frame support, we need to change the code here
                
                pixel_values_tmp = pixel_values_tmp.view(BS, NP, C, H, W)

                if pixel_values_tmp.dim() == 5:
                    pixel_values_tmp = pixel_values_tmp.reshape(BS*NP, C, H, W)
                elif pixel_values_tmp.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(f"pixel_values of shape {pixel_values_tmp.shape}, expect to be of 4 or 5 dimensions")
                
                image_features = image_encoder.model.extract_feature(pixel_values_tmp)
                image_features = image_features.reshape(-1, C_embed)
                                    
                all_image_features.append(image_features)

            vit_embeds = torch.cat(all_image_features, dim=0)
            inputs_embeds_work = inputs_embeds_work.reshape(BS * N_embed, C_embed)
            input_ids = input_ids.reshape(BS * N_embed)
            selected = (input_ids == image_encoder.img_context_token_id)
            n_token = int(selected.sum().item())
            vit_embeds = vit_embeds.reshape(-1, C_embed).to(
                device=inputs_embeds_work.device,
                dtype=inputs_embeds_work.dtype,
            )
            if vit_embeds.size(0) < n_token:
                raise RuntimeError(
                    f"Not enough vision tokens for <IMG_CONTEXT>: required={n_token}, got={vit_embeds.size(0)}"
                )
            if vit_embeds.size(0) > n_token:
                print(
                    f"warning: vision token count mismatch, required={n_token}, got={vit_embeds.size(0)}; truncating."
                )
            with torch.autocast(device_type=inputs_embeds_work.device.type, enabled=False):
                inputs_embeds_work[selected] = vit_embeds[:n_token]
            inputs_embeds_work = inputs_embeds_work.reshape(BS, N_embed, C_embed)
            input_ids = input_ids.reshape(BS, N_embed)
        # pixel_values is not None but is empty ---> text only cases
        elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
            # there are no images
            pass
        
        adaptor_dict['language_inputs'] = inputs_embeds_work.to(dtype=inputs_embeds_dtype)
        adaptor_dict = refresh_adaptor_inputs(adaptor_dict)

    return adaptor_dict
