from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_config: dict = {
    "n_layers": 28,
    "vocab_size": 151936,
    "embedding_dim": 1024,
    "epsilon": 1e-6,
    "head_dim": 128,
    "num_attention_heads": 16,
    "intermediate_size": 3072,
    "context_length": 40960,
    "n_groups": 8,
    "dtype": torch.bfloat16,
    "theta_base": 1000000,
}


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = torch.nn.Linear(
            config["embedding_dim"],
            config["intermediate_size"],
            dtype=config["dtype"],
            bias=False,
        )
        self.down_proj = torch.nn.Linear(
            config["intermediate_size"],
            config["embedding_dim"],
            dtype=config["dtype"],
            bias=False,
        )
        self.gate_proj = torch.nn.Linear(
            config["embedding_dim"],
            config["intermediate_size"],
            dtype=config["dtype"],
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class GQA(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["num_attention_heads"]
        self.n_groups = config["n_groups"]
        self.embedding_dim = config["embedding_dim"]
        self.group_size = self.n_heads // self.n_groups
        self.head_dim = config["head_dim"]
        self.dtype = config["dtype"]

        self.k_proj = torch.nn.Linear(
            self.embedding_dim,
            self.head_dim * self.n_groups,
            bias=False,
            dtype=self.dtype,
        )
        self.v_proj = torch.nn.Linear(
            self.embedding_dim,
            self.head_dim * self.n_groups,
            bias=False,
            dtype=self.dtype,
        )
        self.q_proj = torch.nn.Linear(
            self.embedding_dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype,
        )
        self.o_proj = torch.nn.Linear(
            self.n_heads * self.head_dim,
            self.embedding_dim,
            bias=False,
            dtype=self.dtype,
        )
        self.q_norm = RMSNorm(
            self.head_dim, dtype=self.dtype, epsilon=config["epsilon"]
        )
        self.k_norm = RMSNorm(
            self.head_dim, dtype=self.dtype, epsilon=config["epsilon"]
        )

    def forward(self, x: torch.Tensor, mask, cos, sin):
        b, t, c = x.shape

        k = self.k_proj(x)  # (b, t, self.n_groups * self.head_dim)
        q = self.q_proj(
            x
        )  # (b, t, self.n_heads <-> self.group_size * self.n_groups * self.head_dim)
        v = self.v_proj(x)  # (b, t, self.n_groups * self.head_dim)

        k = k.view(b, t, self.n_groups, self.head_dim).transpose(1, 2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_groups, self.head_dim).transpose(1, 2)

        # normalizing keys and queries
        k = self.k_norm(k)
        q = self.q_norm(q)

        # apply rope
        k = apply_rope(k, cos, sin)
        q = apply_rope(q, cos, sin)

        # Expanding keys and values
        k = k.repeat_interleave(
            self.group_size, dim=1
        )  # In dimension 1 repeat the current state self.group_size times so we can perform everything that we need to
        v = v.repeat_interleave(self.group_size, dim=1)

        att_scores = q @ k.transpose(2, 3)  # att_scores = (b, n_heads, t, t)
        att_masked = torch.masked_fill(att_scores, mask, float("-inf"))
        att_norm = torch.nn.functional.softmax(att_masked / self.head_dim**0.5, dim=-1)

        y = att_norm @ v  # y = (b, n_heads, t, head_dim)
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        y = self.o_proj(y)

        return y


class RMSNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int, dtype: torch.dtype, epsilon: float = 1e-5):
        super().__init__()
        self.dtype = dtype
        self.epsilon = epsilon
        self.weight = torch.nn.Parameter(torch.ones(embedding_dim, dtype=self.dtype))

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=torch.float32)
        rrms = 1 / torch.sqrt(
            torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.epsilon
        )
        x_norm = x * rrms
        return x_norm.to(self.dtype) * self.weight


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(
            config["embedding_dim"], dtype=config["dtype"], epsilon=config["epsilon"]
        )
        self.self_attn = GQA(config)
        self.post_attention_layernorm = RMSNorm(
            config["embedding_dim"], dtype=config["dtype"], epsilon=config["epsilon"]
        )
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), mask, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def compute_rope(config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function will generate cos and sine value tables (rank 2 tensors) that will be used in GQA
    """
    theta = config["theta_base"] ** (
        torch.arange(0, config["head_dim"], 2).float() / config["head_dim"]
    )  # rotation frequency, based on the RoPE paper I will compute  d/2 of these
    inv_freq = 1 / theta  # (head_dim // 2,)
    positions = torch.arange(config["context_length"])  # (context_length,)
    inv_freq = inv_freq.unsqueeze(0)  # (1, head_dim // 2)
    positions = positions.unsqueeze(1)  # (context_lenght, 1)
    angles = positions * inv_freq  # (context_length, head_dim // 2)
    angles = torch.cat([angles, angles], dim=1)  # [[1, 2, 1, 2], [3, 4, 3, 4]]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    b, n_head, t, head_dim = x.shape

    # Splitting x into two halves
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:t, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, t, head_dim)
    sin = sin[:t, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([-x2, x1], dim=-1)
    x_rotated = (x * cos) + (rotated * sin)  # (b, n_head, t, head_dim)

    return x_rotated.to(dtype=x.dtype)


class Qwen3(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config["vocab_size"], config["embedding_dim"], dtype=config["dtype"]
        )
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(config) for i in range(config["n_layers"])]
        )
        self.norm = RMSNorm(
            config["embedding_dim"], config["dtype"], epsilon=config["epsilon"]
        )
        # Tied embeddings
        self.lm_head = torch.nn.Linear(
            config["embedding_dim"], config["vocab_size"], bias=False
        )
        self.lm_head.weight = self.embed_tokens.weight

        cos, sin = compute_rope(config)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor):
        b, t = x.shape
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x, mask, self.cos, self.sin)
        x = self.norm(x)
        x = self.lm_head(x)
        return x  # These are logits at the moment

    @classmethod
    def from_pretrained(cls, model_type: str):
        print(f"Loading weights from pretrained {model_type}")

        config = model_config

        model = Qwen3(config)

        sd = model.state_dict()
        sd_keys = sd.keys()

        # init a huggingface/transformers model
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        sd_hf = {k.removeprefix("model."): v for k, v in model_hf.state_dict().items()}
        sd_hf.pop("lm_head.weight", None)
        for k, v in sd_hf.items():
            assert sd[k].shape == v.shape, f"{k}: {sd[k].shape} != {v.shape}"
            with torch.no_grad():
                sd[k].copy_(v)

        return model


# GREEDY SAMPLING - NOT RECOMMENDED FOR QWEN
def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]  # FOR EACH BATCH GIVE ME
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


if __name__ == "__main__":
    model = Qwen3.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids = torch.tensor(
        [
            tokenizer.encode(
                "<|im_start|>user\nGive me a short introduction to large language models.<|im_end|>\n<|im_start|>assistant\n"
            )
        ]
    )

    generated_tokens = 0
    start = time.time()
    for token in generate_text_basic_stream(
        model=model,
        token_ids=ids,
        max_new_tokens=500,
        eos_token_id=tokenizer.eos_token_id,
    ):
        generated_tokens += 1
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True,  # So the print does not buffer anything but immedietly prints to console
        )
    end = time.time()
    print(f"Total execution time: {end - start}")
