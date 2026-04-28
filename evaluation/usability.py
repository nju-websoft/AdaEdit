import tempfile
import os
import io
import re
import sys
from typing import Tuple
from argparse import ArgumentParser

from pylint.lint import Run
from pylint.reporters.text import TextReporter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import resolve_datasets, get_evaluator_class


def check_for_errors(code_string: str) -> Tuple[bool, str]:
    """
    do not care code style
    """
    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(code_string)
            temp_file.flush()

        report_stream = io.StringIO()
        reporter = TextReporter(report_stream)

        # pylint
        pylint_options = [
            '--errors-only',  # only report errors
            temp_file_name
        ]
        run_result = Run(pylint_options, reporter=reporter, exit=False)
        
        # get result
        stats = run_result.linter.stats
        is_error_free = stats.error == 0 and stats.fatal == 0
        report = report_stream.getvalue().strip()
        
        return is_error_free, report

    finally:
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def ensure_newline(text: str) -> str:
    # Normalize line breaks to Unix style
    text = re.sub(r'\r\n?|\v|\f', '\n', text)
    if text and not text.endswith('\n'):
        text += '\n'
    return text

def check_solutions(evaluator):
    task_ids, predictions, _ = evaluator.process_test_code()
    starter_code_dict = evaluator.get_starter_code()

    count = 0
    unchanged_count = 0
    error_count = 0
    for i, problem in enumerate(predictions):
        task_id = task_ids[i]
        starter_code = ensure_newline(starter_code_dict[task_id])
        for solution in problem:
            count += 1
            if ensure_newline(solution) == starter_code:
                unchanged_count += 1
            else:
                is_error_free, _ = check_for_errors(solution)
                if not is_error_free:
                    error_count += 1
    
    usage_ratio = round((count - unchanged_count - error_count) / count * 100, 2)
    print(f"    {count} solutions processed")
    print(f"    {unchanged_count} solutions have no changes")
    print(f"    {error_count} solutions have linter errors")
    print(f"    {usage_ratio}% solutions are usable")
    
    return usage_ratio


def main():
    parser = ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets')
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    save_dir = args.save_dir

    ds_mappings = resolve_datasets(args.datasets)

    check_res = {}
    for dataset, param_config in ds_mappings.items():
        params_str = f'temp_{param_config["temperature"]}_n_{param_config["n_samples"]}'
        ds_dir = os.path.join(save_dir, params_str, dataset)

        evaluator_class = get_evaluator_class(dataset)
        evaluator = evaluator_class(dataset, ds_dir)

        print(f"\n# {dataset}")
        check_res[dataset] = check_solutions(evaluator)
        
        # # We ignore the second try for aider since it is not a fair comparison
        # if dataset == 'aider':
        #     print('\n# aider_1')
        #     evaluator.try_nums = 1
        #     check_res['aider_1'] = check_solutions(evaluator)
    
    print("\n# Average")
    avg = round(sum(check_res.values()) / len(check_res), 2)
    print(avg)
    

if __name__ == "__main__":
    main()
