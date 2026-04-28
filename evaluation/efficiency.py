import os
import sys
import json
import gzip
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import resolve_datasets, get_evaluator_class


def calculate_canitedit(details_dir: Path, task_id_set) -> dict:
    """
    Analyze the subset of CanitEdit
    """
    def gunzip_json(path: Path):
        """
        Reads a .json.gz file, but produces None if any error occurs.
        """
        try:
            with gzip.open(path, "rt") as f:
                return json.load(f)
        except Exception:
            return None

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    pass_dict = {}
    for fpath in Path(details_dir).glob("*.results.json.gz"):
        record = gunzip_json(fpath)
        task_id = fpath.name.removesuffix(".results.json.gz")
        if task_id not in task_id_set:
            continue
        
        # calulate pass@k
        n = len(record["results"])
        report_k = [x for x in [1, 5, 10, 20] if x <= n]
        c = len([True for r in record["results"] if r["status"] == "OK" and r["exit_code"] == 0])

        for k in report_k:
            if k not in pass_dict:
                pass_dict[k] = []
            pass_dict[k].append(estimator(n, c, k) * 100)
    
    ret = {
        'details': pass_dict,
        "pass@k": {f"pass@{k}": np.mean(pass_dict[k]) for k in pass_dict}
    }

    return ret


def stat_generations(tokenizer, evaluator, output_type, macro, min_tokens, max_tokens):
    # Filter tasks by token range if not in macro mode
    retain_tasks = None
    if not macro:
        starter_code_tokens = {}
        starter_code_dict = evaluator.get_starter_code()
        for task_id, starter_code in starter_code_dict.items():
            starter_code_tokens[task_id] = len(tokenizer.encode(starter_code, add_special_tokens=False))
        # tasks that have token counts within the specified range
        retain_tasks = {k for k, v in starter_code_tokens.items() if v >= min_tokens and v < max_tokens}

        if len(retain_tasks) < len(starter_code_dict) and evaluator.dataset_name == 'canitedit':
            accuracy = calculate_canitedit(os.path.join(evaluator.save_dir, "outputs"), retain_tasks)
            print(f'CanitEdit Accuracy: {accuracy["pass@k"]}')
    
    raw_solutions = []
    with open(evaluator.raw_file, 'r') as f:
        for line in f.readlines():
            item = json.loads(line.strip())
            # Only filter by task_id if not in macro mode
            if retain_tasks is None or item['task_id'] in retain_tasks:
                raw_solutions.extend(item['raw_solutions'])
    
    diff_count = 0
    latency_solutions = []
    if 'fullcode' == output_type:
        # require complete generation
        latency_solutions = raw_solutions
    elif 'diff' == output_type:
        diff_count = len(raw_solutions)
        latency_solutions = [extract_first_hunk(x) for x in raw_solutions]
    elif 'adaptive' in output_type:
        diff_count = sum(1 for item in raw_solutions if item.startswith('diff'))
        latency_solutions = [extract_first_hunk(x) if x.startswith('diff') else x for x in raw_solutions]
    else:
        raise ValueError(f"Output format {output_type} not supported")
    
    cost_counts = analyze_sequence_lengths(tokenizer, raw_solutions)
    latency_counts = analyze_sequence_lengths(tokenizer, latency_solutions)
    
    return diff_count, cost_counts, latency_counts
    

def analyze_sequence_lengths(tokenizer, text_list):
    """ 
    Analyze token lengths for input code 
    """
    ret = []
    for code in text_list:
        tokens = tokenizer.encode(code, add_special_tokens=False)
        ret.append(len(tokens))
    return ret
    

def print_statistics(token_list):
    """
    Print statistics results: mean, min, max, std; 
    
    Args:
        token_list: List of token counts
    """
    if not token_list:
        print("  No data to analyze")
        return
    
    # Basic statistics
    mean_tokens = np.mean(token_list)
    min_tokens = np.min(token_list)
    max_tokens_val = np.max(token_list)
    std_tokens = np.std(token_list)
    total_samples = len(token_list)
    
    print(f"  Total samples: {total_samples}")
    print(f"  Mean tokens: {mean_tokens:.2f}")
    print(f"  Min tokens: {min_tokens}")
    print(f"  Max tokens: {max_tokens_val}")
    print(f"  Std tokens: {std_tokens:.2f}\n")


def extract_first_hunk(diff_str: str):
    """
    Extract the first hunk from a diff string, until the next diff header generated
    """
    SPLIT_HEADER = '\n@@'
    split_info = diff_str.split(SPLIT_HEADER)
    if len(split_info) <= 1:
        return diff_str

    if diff_str.startswith('@@'):
        return split_info[0] + SPLIT_HEADER
    else:
        # startswith diff\n
        if len(split_info) <= 2:
            return diff_str

        return split_info[0] + SPLIT_HEADER + split_info[1] + SPLIT_HEADER


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_type', type=str, choices=['fullcode', 'diff', 'adaptive'])
    parser.add_argument('--macro', action='store_true', help='Calculate macro average across all datasets (ignores min/max_tokens)')
    parser.add_argument('--min_tokens', type=int, default=0)
    parser.add_argument('--max_tokens', type=int, default=int(1e10))
    
    args = parser.parse_args()

    ds_mappings = resolve_datasets(args.datasets)

    save_dir = args.save_dir
    output_type = args.output_type
    macro = args.macro
    min_tokens = args.min_tokens
    max_tokens = args.max_tokens

    # Load tokenizer
    model_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    out_format = {}
    cost_dict = {}
    latency_dict = {}
    for dataset, param_config in ds_mappings.items():
        params_str = f'temp_{param_config["temperature"]}_n_{param_config["n_samples"]}'
        ds_dir = os.path.join(save_dir, params_str, dataset)

        evaluator_class = get_evaluator_class(dataset)
        evaluator = evaluator_class(dataset, ds_dir)

        out_format[dataset], cost_dict[dataset], latency_dict[dataset] = stat_generations(
            tokenizer, evaluator, output_type, macro, min_tokens, max_tokens
        )

        if dataset == 'aider':
            # # We ignore the second try for aider since it is not a fair comparison
            # evaluator.try_nums = 1
            # dataset = 'aider_1'
            # out_format[dataset], cost_dict[dataset], latency_dict[dataset] = stat_generations(
            #     tokenizer, evaluator, output_type, macro, min_tokens, max_tokens
            # )

            if not macro:
                # scale Aider to be comparable
                cost_dict['aider'] *= 20
                latency_dict['aider'] *= 20
                # cost_dict['aider_1'] *= 20
                # latency_dict['aider_1'] *= 20

    
    all_cost = []
    all_latency = []
    for ds_name, cost_list in cost_dict.items():
        print(f'\n# {ds_name}')
        print(f'  {out_format[ds_name]} diff outputs')
        print('Cost tokens:')
        print_statistics(cost_list)
        print('Latency tokens:')
        latency_list = latency_dict[ds_name]
        print_statistics(latency_list)

        if macro:
            # For macro average, collect mean of each dataset
            all_cost.append(np.mean(cost_list))
            all_latency.append(np.mean(latency_list))
        else:
            # For micro average, collect all token counts
            all_cost.extend(cost_list)
            all_latency.extend(latency_list)
    
    if not macro:
        print(f'\nTotal {sum(out_format.values())} diff outputs')
    
    print(f"\n# {'Macro' if macro else 'Micro'} Average")
    print('Cost tokens:')
    print_statistics(all_cost)
    print('Latency tokens:')
    print_statistics(all_latency)


if __name__ == "__main__":
    main()
