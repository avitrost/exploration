#!/usr/bin/env python
"""
Script to perform Best-of-N inference using vLLM on the MATH-500 dataset with Qwen/Qwen2.5-3B.
The script first preprocesses each example by appending an instruction to think step-by-step and
by extracting the ground truth solution. Then for each problem, it generates N responses, verifies
them using Math-Verify, and records the one-indexed position of the first correct answer (or 0 if none).

Usage:
    python best_of_n_math_verify_preprocessed.py --n 1024
"""

import argparse
import csv
import re
from datasets import load_dataset
from tqdm import tqdm

# Import vLLM engine
from vllm import LLM, SamplingParams

# Import the verifier from Math-Verify repository.
# Adjust the import below based on the actual module/function names in the repo.

# Global variable for data source.
data_source = "Maxwell-Jia/AIME_2024"

# Instruction to be appended to each problem.
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def is_correct(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception as e:
        pass

    return ret_score

def is_equiv(output_1: str, output_2: str) -> bool:
    """
    Check if two outputs are equivalent using Math-Verify.
    """
    verify_func = math_metric(
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    try:
        ret_score_1, _ = verify_func([output_1], [output_2])
        # ret_score_2, _ = verify_func([output_2], [output_1])
        ret_score = ret_score_1 #or ret_score_2
    except Exception as e:
        pass

    return ret_score

# from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
import numpy as np
from collections import Counter


# def extract_solution(solution_str):
#     print(solution_str)
#     print(last_boxed_only_string(solution_str))
#     print(remove_boxed(last_boxed_only_string(solution_str)))
#     return remove_boxed(last_boxed_only_string(solution_str))


# Preprocessing function to add a unique id and structure each data item.
def make_map_fn(split):
    def process_fn(example, idx):
        # Remove the 'problem' field and append the instruction.
        question = example.pop('Problem')
        question = question + ' ' + instruction_following

        # Remove the 'solution' field and extract the ground truth.
        solution = str(example.pop('Answer'))
        # solution = extract_solution(answer)

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx
            },
            "unique_id": f"{data_source}-{split}-{idx}"
        }
        return data
    return process_fn

def generate_responses(engine, prompt, max_new_tokens, num_return_sequences):
    """
    Generate a list of responses using the vLLM engine.
    
    Parameters:
        engine: vLLM engine instance.
        prompt (str): The text prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        num_return_sequences (int): Number of responses to generate.
        
    Returns:
        List of generated text responses.
    """
    generation_outputs = engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        # Add additional generation parameters (e.g., temperature, top_p) if needed.
    )
    # Adjust extraction based on the vLLM output structure.
    responses = [gen.output_text for gen in generation_outputs]
    return responses

def main():
    parser = argparse.ArgumentParser(description="Best-of-N Inference with Math-Verify and Preprocessing")
    parser.add_argument("--n", type=int, default=1024, help="Number of responses to generate per problem")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Output CSV file to record results")
    args = parser.parse_args()

    # Load the MATH-500 dataset (using the test split; adjust if needed)
    print("Loading Maxwell-Jia/AIME_2024 dataset...")
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    # Preprocess the dataset by mapping each example.
    print("Preprocessing dataset...")
    processed_dataset = dataset.map(make_map_fn("train"), with_indices=True)

    # Initialize the vLLM engine for Qwen/Qwen2.5-3B.
    print("Initializing vLLM engine for Qwen/Qwen3-1.7B...")
    model_id = "Qwen/Qwen3-1.7B"
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=1,  # Adjust based on your GPU count
        gpu_memory_utilization=0.8,
        trust_remote_code=True   # Required for Qwen models
    )

    sampling_params = SamplingParams(
        temperature=1,
        top_p=1,
        max_tokens=1024,
        n=1
    )


    results = []
    total = len(processed_dataset)
    for idx, example in enumerate(processed_dataset):
        # Extract the prompt text from the preprocessed example.
        prompt_text = example["prompt"][0]["content"]
        ground_truth = example["reward_model"]["ground_truth"]

        print(f"\nProcessing problem {idx+1}/{total} (Unique index: {example['extra_info']['index']})")
        print("Prompt:", prompt_text)

        # Generate N responses for the current problem.
        print(f"Generating {args.n} responses...")
        outputs = llm.generate([prompt_text] * 1024, sampling_params)


        # Use Math-Verify to check each generated response.
        first_correct = 0  # default: no correct response found
        unique_outputs = set()  # To track unique outputs
        for gen_idx, output in tqdm(enumerate(outputs)):
            output_text = output.outputs[0].text
            if first_correct == 0 and is_correct(output_text, ground_truth):
                first_correct = gen_idx + 1  # record one-indexed position
                print(f"Found correct answer at response {first_correct}")
                # break
            # Store unique outputs
            novel_answer = True
            for provided_answer in unique_outputs:
                if is_equiv(output_text, provided_answer):
                    # print(f"Response {gen_idx + 1} is equivalent to a previous response.")
                    novel_answer = False
                    break
            if novel_answer:
                unique_outputs.add(output_text)
                # print(f"Novel response {gen_idx + 1}: {output_text}")

        if first_correct == 0:
            print("No correct answer found among the generated responses.")

        results.append({
            "data_source": example["data_source"],
            "split": example["extra_info"]["split"],
            "index": example["extra_info"]["index"],
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "first_correct_index": first_correct,
            "unique_outputs": list(unique_outputs),
            "num_unique_outputs": len(unique_outputs)
        })

    # Count occurrences of each first_correct_index value
    
    # Find the maximum correct index to determine array size
    max_index = 1024
    
    # Create array with size max_index + 1 (to include 0 index for "none correct")
    first_correct_counts = np.zeros(max_index + 1, dtype=int)
    
    # Count each occurrence
    for result in results:
        first_correct_counts[result["first_correct_index"]] += 1

    unique_answers_counts = [result["num_unique_outputs"] for result in results]
    print(f"\nCounts of unique answers per problem: {unique_answers_counts}")
    # Calculate statistics of unique answers counts
    mean_unique = np.mean(unique_answers_counts)
    median_unique = np.median(unique_answers_counts)
    min_unique = np.min(unique_answers_counts)
    max_unique = np.max(unique_answers_counts)
    std_unique = np.std(unique_answers_counts)

    print(f"\nUnique answer statistics:")
    print(f"Mean: {mean_unique:.2f}")
    print(f"Median: {median_unique:.2f}")
    print(f"Min: {min_unique}")
    print(f"Max: {max_unique}")
    print(f"Standard Deviation: {std_unique:.2f}")

    # Create a histogram of unique answers count
    unique_counts_histogram = Counter(unique_answers_counts)
    print("\nHistogram of unique answers per problem:")
    for count in sorted(unique_counts_histogram.keys()):
        print(f"{count} unique answers: {unique_counts_histogram[count]} problems")
    
    print(f"\nCounts of first correct answer by position (0 = none correct):")
    print(first_correct_counts)
    # Save the first_correct_counts array to a file
    np.save(f"first_correct_counts_{args.n}_aime.npy", first_correct_counts)
    print(f"Saved first_correct_counts to first_correct_counts_{args.n}.npy")

    # Calculate and print some useful metrics
    num_problems = len(results)
    num_solved = num_problems - first_correct_counts[0]
    print(f"Solved {num_solved}/{num_problems} problems ({num_solved/num_problems:.2%})")
    
    # Still write the detailed results to CSV
    print(f"\nWriting results to {args.output_csv} ...")
    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["data_source", "split", "index", "prompt", "ground_truth", "first_correct_index", "unique_outputs", "num_unique_outputs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Done.")

if __name__ == "__main__":
    main()
