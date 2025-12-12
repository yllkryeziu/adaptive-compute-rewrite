#!/usr/bin/env python3
"""Convert ShareGPT-style preference data to NeMo-RL DPO format.

Input format (LLaMA-Factory ShareGPT):
{
  "conversations": [
    {"from": "system", "value": "..."},
    {"from": "human", "value": "..."}
  ],
  "chosen": {"from": "gpt", "value": "..."},
  "rejected": {"from": "gpt", "value": "..."}
}

Output format (NeMo-RL BinaryPreferenceDataset):
{
  "prompt": "<formatted prompt with chat template>",
  "chosen": "<preferred response>",
  "rejected": "<non-preferred response>"
}
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer


def format_prompt_qwen(conversations: list[dict]) -> str:
    """Format conversations into Qwen chat template format."""
    formatted_parts = []

    for msg in conversations:
        role = msg["from"]
        content = msg["value"]

        if role == "system":
            formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "human":
            formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "gpt":
            formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    # Add the assistant start token for the response
    formatted_parts.append("<|im_start|>assistant\n")

    return "\n".join(formatted_parts)


def format_prompt_with_tokenizer(
    conversations: list[dict],
    tokenizer: AutoTokenizer,
) -> str:
    """Format conversations using the tokenizer's chat template."""
    messages = []

    for msg in conversations:
        role = msg["from"]
        content = msg["value"]

        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})

    # Apply chat template without generating the assistant response
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def convert_dataset(
    input_path: str,
    output_path: str,
    model_name: Optional[str] = None,
    use_tokenizer: bool = False,
    max_samples: Optional[int] = None,
) -> None:
    """Convert ShareGPT preference data to NeMo-RL format.

    Args:
        input_path: Path to input JSON file (ShareGPT format)
        output_path: Path to output JSONL file (NeMo-RL format)
        model_name: Model name for tokenizer (if use_tokenizer=True)
        use_tokenizer: Whether to use tokenizer's chat template
        max_samples: Maximum number of samples to convert
    """
    print(f"Loading data from {input_path}")
    with open(input_path, "r") as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    tokenizer = None
    if use_tokenizer and model_name:
        print(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    converted = []
    skipped = 0

    for i, item in enumerate(data):
        try:
            conversations = item["conversations"]
            chosen_response = item["chosen"]["value"]
            rejected_response = item["rejected"]["value"]

            # Format the prompt
            if tokenizer and use_tokenizer:
                prompt = format_prompt_with_tokenizer(conversations, tokenizer)
            else:
                prompt = format_prompt_qwen(conversations)

            converted.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            })

            if (i + 1) % 1000 == 0:
                print(f"Converted {i + 1}/{len(data)} samples")

        except KeyError as e:
            skipped += 1
            if skipped <= 5:
                print(f"Warning: Skipping sample {i} due to missing key: {e}")

    # Write output as JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(converted)} samples to {output_path}")
    with open(output_path, "w") as f:
        for item in converted:
            f.write(json.dumps(item) + "\n")

    print(f"Done! Converted {len(converted)} samples, skipped {skipped}")

    # Print sample for verification
    if converted:
        print("\n--- Sample output ---")
        sample = converted[0]
        print(f"Prompt (first 500 chars): {sample['prompt'][:500]}...")
        print(f"Chosen (first 200 chars): {sample['chosen'][:200]}...")
        print(f"Rejected (first 200 chars): {sample['rejected'][:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT preference data to NeMo-RL DPO format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/simpo/Sky-T1_preference_data_10k.json",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/p/project1/envcomp/yll/adaptive-compute-rewrite/data/nemo_simpo_data.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for tokenizer chat template",
    )
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="Use tokenizer's chat template instead of manual formatting",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert",
    )

    args = parser.parse_args()

    convert_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        use_tokenizer=args.use_tokenizer,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
