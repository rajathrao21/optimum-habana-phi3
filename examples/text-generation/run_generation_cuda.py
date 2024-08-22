import os, time, argparse
import torch
from itertools import cycle
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_args(parser):
    parser.add_argument("--device", "-d", type=str, choices=["cuda"], help="Device to run", default="cuda")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Whether to perform generation in float16 (half) precision.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature value for text generation")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top_p value for generating text via sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    return parser.parse_args()


def book_dataset(args):
    def download_book(book_id):
        import os
        import requests

        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        response = requests.get(url)
        if response.status_code == 200:
            pid = os.getpid()
            save_path = f"/tmp/{book_id}_{pid}.txt"
            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"Book downloaded and saved to: {save_path}")
            return save_path
        else:
            print("Failed to download book! Exiting...")
            import sys

            sys.exit()
    
    def assemble_prompt(prompt_size, book_path, batch_size):
        prompt = ""
        counter = 0
        book_lines = open(book_path).readlines()
        for line in book_lines:
            for word in line.split():
                counter += 1
                prompt += word + " "
                if counter == prompt_size:
                    return [prompt] * batch_size
    
    book_ids = [
        2701,  # Moby Dick; Or, The Whale
        1513,  # Romeo and Juliet
        1342,  # Pride and Prejudice
    ]
    return assemble_prompt(prompt_size=args.max_input_tokens, book_path=download_book(book_ids[0]), batch_size=args.batch_size)


def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    max_length = args.max_input_tokens + args.max_new_tokens
    config = AutoConfig.from_pretrained(args.model_name_or_path, max_length=max_length)
    model = AutoModelForCausalLM.from_config(config)
    if args.half:
        model = model.to(dtype=torch.float16, device=args.device)
    elif args.bf16:
        model = model.to(dtype=torch.bfloat16, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    input_sentences = book_dataset(args)
    if args.batch_size > len(input_sentences):
        # Dynamically extends to support larger batch sizes
        num_sentences_to_add = args.batch_size - len(input_sentences)
        for i in range(num_sentences_to_add):
            input_sentences.append(input_sentences[i % len(input_sentences)])
    elif args.batch_size < len(input_sentences):
        input_sentences = input_sentences[: args.batch_size]

    def generate(reduce_recompile=False):
        # Tokenization
        if args.max_input_tokens > 0:
            input_tokens = tokenizer.batch_encode_plus(
                input_sentences,
                return_tensors="pt",
                padding="max_length",
                max_length=args.max_input_tokens,
                truncation=True,
            )
        else:
            input_tokens = tokenizer.batch_encode_plus(input_sentences, return_tensors="pt", padding=True)

        if args.half:
            input_tokens = {key: value.to(dtype=torch.float16, device=args.device) for key, value in input_tokens.items()}
        elif args.bf16:
            input_tokens = {key: value.to(dtype=torch.bfloat16, device=args.device) for key, value in input_tokens.items()}

        if not reduce_recompile:
            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(args.device)

        # Generate with max_new_tokens instead of max_length
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
        ).cpu()

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    total_new_tokens_generated = 0
    t0 = time.perf_counter()
    for i in range(args.n_iterations):
        generated = generate()
    duration = time.perf_counter() - t0
    total_new_tokens_generated = args.n_iterations * args.batch_size * args.max_new_tokens
    throughput = total_new_tokens_generated / duration

    print()
    print("Input/outputs:")
    for i, input_sentence in enumerate(zip(input_sentences)):
        print(f"input {i+1}: {input_sentence}")
        for j, output in enumerate(
            zip(generated[args.num_return_sequences * i : args.num_return_sequences * (i + 1)])
        ):
            print(f"output {j+1}: {output}")
        print()

    print(f"Throughput (including tokenization) = {throughput} tokens/second")


if __name__ == "__main__":
    main()
