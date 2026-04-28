import os
import sys
import json
from typing import Any
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import initialize_diff_processor, resolve_datasets, get_evaluator_class


def check_patch(diff_tool, evaluator, output_type):
    '''
    Check the patch correctness.
    '''
    info = []

    starter_code_dict = evaluator.get_starter_code()
    with open(evaluator.raw_file, 'r') as f:
        for line in f.readlines():
            item = json.loads(line.strip())
            starter_code = starter_code_dict[item['task_id']]
            if output_type == 'adaptive':
                for raw_solution in item['raw_solutions']:
                    split_lines = raw_solution.splitlines(keepends=True)
                    content = raw_solution[len(split_lines[0]):]
                    if split_lines[0].strip() == 'diff':
                        try:
                            diff_tool.apply_diff(starter_code, content)
                        except Exception:
                            info.append(-1)
                        else:
                            info.append(1)
                    else:
                        # full code
                        info.append(0)
            else:
                for raw_solution in item['raw_solutions']:
                    try:
                        diff_tool.apply_diff(starter_code, raw_solution)
                    except Exception:
                        info.append(-1)
                    else:
                        info.append(1)
    
    return info
    

def statistics(check_info):
    ret = {}

    diff_count = sum(1 for item in check_info if item != 0)
    error_count = sum(1 for item in check_info if item == -1)

    total_count = len(check_info)
    ret['count'] = total_count
    ret['diff_count'] = diff_count
    ret['diff_error'] = error_count
    ret['accuracy'] = round((total_count - error_count) / total_count * 100, 2)

    print_statistics(ret)

    return ret


def print_statistics(stat_info):
    print(f'Count: {stat_info.get("count")}')
    print(f'Diff count: {stat_info.get("diff_count")}')
    print(f'Diff error: {stat_info.get("diff_error")}')
    print(f'Accuracy: {stat_info.get("accuracy")}')


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets')
    parser.add_argument('--edit_format', type=str, required=True)
    parser.add_argument('--output_type', type=str)

    args = parser.parse_args()

    save_dir = args.save_dir
    edit_format = args.edit_format
    output_type = args.output_type

    assert os.path.isdir(save_dir)

    # prepare tool for diff and patch
    diff_tool, output_type, _ = initialize_diff_processor(edit_format, output_type)
    assert diff_tool, "Diff tool not found"
    diff_tool.strict_mode = True

    ds_mappings = resolve_datasets(args.datasets)
    
    check_res = {}
    for dataset, param_config in ds_mappings.items():
        params_str = f'temp_{param_config["temperature"]}_n_{param_config["n_samples"]}'
        ds_dir = os.path.join(save_dir, params_str, dataset)

        evaluator_class = get_evaluator_class(dataset)
        evaluator = evaluator_class(dataset, ds_dir)
        check_res[dataset] = check_patch(diff_tool, evaluator, output_type)

        if dataset == 'aider':
            # scale Aider to be comparable
            check_res[dataset] *= 20

            # # We ignore the second try for aider since it is not a fair comparison
            # evaluator.try_nums = 1
            # dataset = 'aider_1'
            # check_res[dataset] = check_patch(diff_tool, evaluator, output_type) * 20
    
    sum_dict = {}
    for ds_name, check_info in check_res.items():
        print(f"Dataset: {ds_name}")
        range_info = statistics(check_info)
        for key, value in range_info.items():
            if key in sum_dict:
                sum_dict[key] += value
            else:
                sum_dict[key] = value
    
    print("All datasets combined")
    sum_dict = {key: round(value / len(check_res), 2) for key, value in sum_dict.items()}
    print_statistics(sum_dict)

    print('Micro-average')
    all_check_res = []
    for check_info in check_res.values():
        all_check_res += check_info
    statistics(all_check_res)
    

if __name__ == "__main__":
    main()
