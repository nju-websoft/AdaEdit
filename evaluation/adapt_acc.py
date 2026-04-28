import os
import sys
import json
import re

from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import initialize_diff_processor, resolve_datasets, get_evaluator_class


def _extract_first_fenced_block(text):
    if not isinstance(text, str) or not text:
        return "", ""

    block_match = re.search(r"```([^\n`]*)\n([\s\S]*?)```", text)
    if block_match:
        return block_match.group(1).strip().lower(), block_match.group(2)

    open_match = re.search(r"```([^\n`]*)\n([\s\S]*)$", text)
    if open_match:
        return open_match.group(1).strip().lower(), open_match.group(2)

    return "", text


def _normalize_raw_solution(raw_solution, backend_mode):
    if not isinstance(raw_solution, str):
        raw_solution = "" if raw_solution is None else str(raw_solution)

    if backend_mode == "api":
        if "```" in raw_solution:
            tag, body = _extract_first_fenced_block(raw_solution)
            normalized_body = body.strip("\n") if body else ""
            return f"{tag}\n{normalized_body}"

    return raw_solution


def _parse_solution(raw_solution, backend_mode):
    normalized = _normalize_raw_solution(raw_solution, backend_mode)
    if not normalized.strip():
        return None, None, True

    split_lines = normalized.splitlines(keepends=True)
    if not split_lines:
        return None, None, True

    raw_use_diff = split_lines[0].strip() == "diff"
    content = normalized[len(split_lines[0]):]
    if not content.strip():
        return raw_use_diff, None, True

    return raw_use_diff, content, False


def check_format(tokenizer, diff_tool, evaluator, backend_mode):
    '''
    Check the correctness and bias of the generated format.
    '''
    ret = []

    starter_code_dict = evaluator.get_starter_code()
    with open(evaluator.raw_file, 'r') as f:
        for line in f.readlines():
            item = json.loads(line.strip())
            task_id = item['task_id']
            starter_code = diff_tool._ensure_newline(starter_code_dict[task_id])
            for raw_solution in item['raw_solutions']:
                raw_use_diff, content, is_no_change = _parse_solution(raw_solution, backend_mode)
                if is_no_change:
                    ret.append({'diff_error': True, 'raw_use_diff': raw_use_diff})
                    continue

                use_diff = raw_use_diff
                if use_diff:
                    # select diff
                    diff = content
                    try:
                        code = diff_tool.apply_diff(starter_code, content)
                    except Exception:
                        ret.append({'diff_error': True, 'raw_use_diff': raw_use_diff})
                        continue
                else:
                    # select fullcode
                    code = content
                    try:
                        diff = diff_tool.calculate_diff(starter_code, code, lang='python')
                    except Exception:
                        ret.append({'diff_error': True, 'raw_use_diff': raw_use_diff})
                        continue
                
                if diff_tool._ensure_newline(code) == starter_code:
                    # No change
                    ret.append({'diff_error': True, 'raw_use_diff': raw_use_diff})
                else:
                    ret_item = {"use_diff": use_diff, "raw_use_diff": raw_use_diff}
                    code_len = len(tokenizer.encode(code, add_special_tokens=False))
                    diff_len = len(tokenizer.encode(diff, add_special_tokens=False))
                    if use_diff == (diff_len < code_len):
                        ret_item['correct'] = True
                    else:
                        # error: calculate bias
                        ret_item['correct'] = False
                        if diff_len < code_len:
                            ret_item['bias'] = (code_len - diff_len) / max(diff_len, 1)
                        else:
                            ret_item['bias'] = (diff_len - code_len) / max(code_len, 1)
                    ret.append(ret_item)
    
    return ret
    

def statistics(check_info):
    total_count = len(check_info)

    valid_info = [x for x in check_info if not x.get('diff_error', False)]
    diff_errors = len(check_info) - len(valid_info)

    raw_diff_count = sum(1 for item in check_info if item.get('raw_use_diff') is True)
    raw_diff_unknown = sum(1 for item in check_info if item.get('raw_use_diff') is None)

    diff_count = sum(1 for item in valid_info if item['use_diff'])

    error_info = [x for x in valid_info if not x['correct']]
    correct_count = len(valid_info) - len(error_info)

    ret = {}
    last_value = 0
    for tolerance in [0.2, 0.5]:
        tolerance = round(tolerance, 1)
        tolerance_count = sum(1 for item in error_info if item['bias'] <= tolerance and item['bias'] > last_value)
        ret[f'<= {tolerance}'] = round(tolerance_count / total_count * 100, 2)
        last_value = tolerance
    
    over_count = sum(1 for item in error_info if item['bias'] > 0.5)
    ret['> 0.5'] = round(over_count / total_count * 100, 2)
    
    ret['raw_diff_count'] = round(raw_diff_count / total_count * 100, 2)
    ret['raw_diff_unknown'] = round(raw_diff_unknown / total_count * 100, 2)
    ret['diff_count'] = round(diff_count / total_count * 100, 2)
    ret['diff_error'] = round(diff_errors / total_count * 100, 2)
    ret['correct'] = round(correct_count / total_count * 100, 2)

    print_statistics(ret)

    return ret


def print_statistics(stat_info):
    stat_info = stat_info.copy()
    print(f'  Raw Diff count: {stat_info.pop("raw_diff_count")}')
    print(f'  Raw Diff unknown: {stat_info.pop("raw_diff_unknown")}')
    print(f'  Diff count: {stat_info.pop("diff_count")}')
    print(f'  No Change: {stat_info.pop("diff_error")}')
    print(f'  Correct: {stat_info.pop("correct")}')
    for tolerance, rate in sorted(stat_info.items()):
        print(f'  Bias {tolerance}: {rate}')


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--edit_format', type=str, required=True)
    parser.add_argument('--backend_mode', choices=['vllm', 'api'], default='vllm')

    args = parser.parse_args()

    save_dir = args.save_dir
    model_path = args.model
    edit_format = args.edit_format
    backend_mode = args.backend_mode

    diff_tool = initialize_diff_processor(edit_format)[0]
    assert diff_tool, 'Failed to initialize diff tool'

    ds_mappings = resolve_datasets(args.datasets)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    format_check_res = {}
    for dataset, param_config in ds_mappings.items():
        if backend_mode == 'api':
            params_str = 'temp_0.0_n_1'
        else:
            params_str = f'temp_{param_config["temperature"]}_n_{param_config["n_samples"]}'
        ds_dir = os.path.join(save_dir, params_str, dataset)

        evaluator_class = get_evaluator_class(dataset)
        evaluator = evaluator_class(dataset, ds_dir)

        format_check_res[dataset] = check_format(tokenizer, diff_tool, evaluator, backend_mode)
        # #  the second try for aider
        # if dataset == 'aider':
        #     evaluator.try_nums = 1
        #     format_check_res['aider_1'] = check_format(tokenizer, diff_tool, evaluator, backend_mode)

    sum_dict = {}
    for ds_name, check_info in format_check_res.items():
        print(f"\n# {ds_name}")
        range_info = statistics(check_info)
        for key, value in range_info.items():
            sum_dict[key] = sum_dict.get(key, 0) + value
    
    print("\n# Average")
    sum_dict = {key: round(value / len(format_check_res), 2) for key, value in sum_dict.items()}
    print_statistics(sum_dict)
    

if __name__ == "__main__":
    main()
