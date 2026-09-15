from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)
            out = out.logits[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
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
