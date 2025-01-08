import copy
import gc
import logging
import queue
import threading

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr

from nanogcg.utils import (
    INIT_CHARS,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_r: Optional[int] = None
    probe_sampling_factor: int = 16
    retry_limit: int = 0


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad.device)
    )[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(
            ids_decoded[i], return_tensors="pt", add_special_tokens=False
        ).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
        draft_model: Optional[transformers.PreTrainedModel],
        draft_tokenizer: Optional[transformers.PreTrainedTokenizer],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )
        self.prefix_cache = None
        self.draft_prefix_cache = None

        self.stop_flag = False

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.draft_embedding_layer = None
        if self.draft_model and self.draft_tokenizer:
            logger.debug("Probe sampling enabled.")
            self.draft_embedding_layer = self.draft_model.get_input_embeddings()
            if self.draft_tokenizer.pad_token is None:
                # TODO document why
                self.draft_tokenizer.pad_token = tokenizer.eos_token
                # self.draft_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            # TODO not sure if needed
            # self.draft_model.eval()

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if model.device == torch.device("cpu"):
            logger.warning(
                "Model is on the CPU. Use a hardware accelerator for faster optimization."
            )

        if not tokenizer.chat_template:
            logger.warning(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device, torch.int64)
        after_ids = tokenizer(
            [after_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [
            embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
        ]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        if self.draft_model and self.draft_tokenizer and self.draft_embedding_layer:
            # Tokenize everything that doesn't get optimized for the draft model, if probe sampling is enabled
            draft_before_ids = self.draft_tokenizer(
                [before_str], padding=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            draft_after_ids = self.draft_tokenizer(
                [after_str], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer(
                [target], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)

            (
                self.draft_before_embeds,
                self.draft_after_embeds,
                self.draft_target_embeds,
            ) = [
                self.draft_embedding_layer(ids)
                for ids in (
                    draft_before_ids,
                    draft_after_ids,
                    self.draft_target_ids,
                )
            ]

            if config.use_prefix_cache:
                with torch.no_grad():
                    output = self.draft_model(
                        inputs_embeds=self.draft_before_embeds, use_cache=True
                    )
                    self.draft_prefix_cache = output.past_key_values

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for _ in tqdm(range(config.num_steps)):
            # Compute the token gradient
            # torch.Size([1, 20, 50257]) for gpt2, the grad if replacing token i with j of the V.
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = (
                    new_search_width if config.batch_size is None else config.batch_size
                )
                if self.prefix_cache:
                    input_embeds = torch.cat(
                        [
                            embedding_layer(sampled_ids),
                            after_embeds.repeat(new_search_width, 1, 1),
                            target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )
                else:
                    input_embeds = torch.cat(
                        [
                            before_embeds.repeat(new_search_width, 1, 1),
                            embedding_layer(sampled_ids),
                            after_embeds.repeat(new_search_width, 1, 1),
                            target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )

                retry_limit = self.config.retry_limit
                trial_count = 0
                while True:
                    if self.draft_model is None:
                        loss = find_executable_batch_size(
                            self._compute_candidates_loss_original, batch_size
                        )(input_embeds)
                    else:
                        loss = find_executable_batch_size(
                            self._compute_candidates_loss_probe_sampling, batch_size
                        )(input_embeds, sampled_ids)

                    current_loss = loss.min().item()
                    optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                    logger.debug(
                        f"Current loss: {current_loss}, buffer highest: {buffer.get_highest_loss()}"
                    )

                    # Update the buffer based on the loss
                    if current_loss < buffer.get_highest_loss():
                        losses.append(current_loss)
                        buffer.add(current_loss, optim_ids)
                        break
                    elif trial_count >= retry_limit:
                        if buffer.size == 0:
                            # TODO: should pick the best one from the retries
                            buffer.add(current_loss, optim_ids)
                        losses.append(current_loss)
                        break
                    else:
                        trial_count += 1
                        logger.info(f"Loss not optimized. Retrying #{trial_count}.")

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = (
                    tokenizer(
                        INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(model.device)
                )
                init_indices = torch.randint(
                    0,
                    init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids = tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model.device)
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat(
                [
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds = torch.cat(
                [
                    self.before_embeds.repeat(true_buffer_size, 1, 1),
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )

        init_buffer_losses = find_executable_batch_size(
            self._compute_candidates_loss_original, true_buffer_size
        )(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat(
                [optim_embeds, self.after_embeds, self.target_embeds], dim=1
            )
            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self.prefix_cache,
                use_cache=True,
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[
            ..., shift - 1 : -1, :
        ].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(
                shift_logits, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[loss], inputs=[optim_ids_onehot]
        )[0]

        return optim_ids_onehot_grad

    def _compute_candidates_loss_probe_sampling(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        sampled_ids: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """

        B = input_embeds.shape[0]
        probe_size = B // self.config.probe_sampling_factor
        probe_idxs = torch.randperm(B)[:probe_size].to(input_embeds.device)
        probe_embeds = input_embeds[probe_idxs]

        # Helper 1 - Will be executed by probe thread below
        def _compute_probe_losses(
            result_queue: queue.Queue, search_batch_size: int, probe_embeds: Tensor
        ) -> None:
            probe_losses = self._compute_candidates_loss_original(
                search_batch_size, probe_embeds
            )
            result_queue.put(("probe", probe_losses))
            logger.debug("Probe thread done.")

        # Will be executed by draft thread below
        def _compute_draft_losses(
            result_queue: queue.Queue,
            search_batch_size: int,
            draft_sampled_ids: Tensor,
        ) -> None:
            assert (
                self.draft_model and self.draft_embedding_layer
            ), "Draft model and embedding layer weren't initialized properly."

            draft_losses = []
            draft_prefix_cache_batch = None
            for i in range(0, B, search_batch_size):
                batch_size = min(search_batch_size, B - i)
                draft_sampled_ids_batch = draft_sampled_ids[i : i + batch_size]

                if self.draft_prefix_cache:
                    if not draft_prefix_cache_batch or batch_size != search_batch_size:
                        draft_prefix_cache_batch = [
                            [
                                x.expand(batch_size, -1, -1, -1)
                                for x in self.draft_prefix_cache[i]
                            ]
                            for i in range(len(self.draft_prefix_cache))
                        ]
                    logger.debug(
                        self.draft_embedding_layer(draft_sampled_ids_batch).shape
                    )
                    logger.debug(draft_sampled_ids_batch.shape)
                    logger.debug(self.draft_after_embeds.shape)
                    logger.debug(self.draft_after_embeds.repeat(batch_size, 1, 1).shape)
                    logger.debug(self.draft_target_embeds.shape)
                    logger.debug(
                        self.draft_target_embeds.repeat(batch_size, 1, 1).shape
                    )
                    draft_embeds = torch.cat(
                        [
                            self.draft_embedding_layer(draft_sampled_ids_batch),
                            self.draft_after_embeds.repeat(batch_size, 1, 1),
                            self.draft_target_embeds.repeat(batch_size, 1, 1),
                        ],
                        dim=1,
                    )
                    draft_output = self.draft_model(
                        inputs_embeds=draft_embeds,
                        past_key_values=draft_prefix_cache_batch,
                    )
                else:
                    draft_embeds = torch.cat(
                        [
                            self.draft_before_embeds.repeat(batch_size, 1, 1),
                            self.draft_embedding_layer(draft_sampled_ids_batch),
                            self.draft_after_embeds.repeat(batch_size, 1, 1),
                            self.draft_target_embeds.repeat(batch_size, 1, 1),
                        ],
                        dim=1,
                    )
                    draft_output = self.draft_model(inputs_embeds=draft_embeds)

                draft_logits = draft_output.logits
                tmp = draft_embeds.shape[1] - self.draft_target_ids.shape[1]
                shift_logits = draft_logits[..., tmp - 1 : -1, :].contiguous()
                shift_labels = self.draft_target_ids.repeat(batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(
                        shift_logits, -1, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    loss = mellowmax(
                        -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
                    )
                else:
                    loss = (
                        torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction="none",
                        )
                        .view(batch_size, -1)
                        .mean(dim=-1)
                    )

                draft_losses.append(loss)

            draft_losses = torch.cat(draft_losses)
            result_queue.put(("draft", draft_losses))
            logger.debug("Draft thread done.")

        def _convert_to_draft_tokens(token_ids: Tensor) -> Tensor:
            decoded_text_list = self.tokenizer.batch_decode(token_ids)
            assert self.draft_tokenizer, "Draft tokenizer wasn't properly initialized."
            return self.draft_tokenizer(
                decoded_text_list,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )["input_ids"].to(self.draft_model.device, torch.int64)

        with torch.no_grad():
            result_queue = queue.Queue()
            draft_sampled_ids = _convert_to_draft_tokens(sampled_ids)

            draft_thread = threading.Thread(
                target=_compute_draft_losses,
                args=(result_queue, search_batch_size, draft_sampled_ids),
            )

            probe_thread = threading.Thread(
                target=_compute_probe_losses,
                args=(result_queue, search_batch_size, probe_embeds),
            )

            draft_thread.start()
            probe_thread.start()

            draft_thread.join()
            probe_thread.join()

        results = {}
        while not result_queue.empty():
            key, value = result_queue.get()
            results[key] = value

        probe_losses = results["probe"]
        draft_losses = results["draft"]

        # Step 4. Calculate agreement score using Spearman correlation
        draft_probe_losses = draft_losses[probe_idxs]
        rank_correlation = spearmanr(
            probe_losses.cpu().type(torch.float32).numpy(),
            draft_probe_losses.cpu().type(torch.float32).numpy(),
        ).correlation
        # normalized from [-1, 1] to [0, 1]
        alpha = (1 + rank_correlation) / 2

        # 5. Filter candidates based on draft model losses
        R = 8 if self.config.probe_sampling_r is None else self.config.probe_sampling_r
        filtered_size = int((1 - alpha) * B / R)
        filtered_size = max(1, min(filtered_size, B))

        _, top_indices = torch.topk(draft_losses, k=filtered_size, largest=False)

        # 6. Compute losses on filtered set with target model
        filtered_embeds = input_embeds[top_indices]
        filtered_losses = self._compute_candidates_loss_original(
            search_batch_size, filtered_embeds
        )

        # 7. Return best loss between probe set and filtered set
        best_probe_loss = probe_losses.min()
        best_filtered_loss = filtered_losses.min()

        logger.debug(f"Correlation: {alpha}")
        logger.debug(f"Filtered size: {filtered_size}")
        logger.debug(f"Probe losses: {probe_losses}")
        logger.debug(f"Draft losses: {draft_losses.shape}")
        logger.debug(f"Draft probe losses: {draft_probe_losses}")
        logger.debug(f"Probe indices: {probe_idxs}")
        logger.debug(f"Top indices: {top_indices}")
        logger.debug(f"Top draft losses: {draft_losses[top_indices]}")
        logger.debug(f"Best probe loss: {best_probe_loss}")
        logger.debug(f"Best filtered loss: {best_filtered_loss}")

        return probe_losses if best_probe_loss < best_filtered_loss else filtered_losses

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if (
                        not prefix_cache_batch
                        or current_batch_size != search_batch_size
                    ):
                        prefix_cache_batch = [
                            [
                                x.expand(current_batch_size, -1, -1, -1)
                                for x in self.prefix_cache[i]
                            ]
                            for i in range(len(self.prefix_cache))
                        ]

                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=prefix_cache_batch,
                        use_cache=True,
                    )
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(
                        shift_logits, -1, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    loss = mellowmax(
                        -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    )

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.config.early_stop:
                    if torch.any(
                        torch.all(
                            torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1
                        )
                    ).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
    draft_model: Optional[transformers.PreTrainedModel] = None,
    draft_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config, draft_model, draft_tokenizer)
    result = gcg.run(messages, target)
    return result
