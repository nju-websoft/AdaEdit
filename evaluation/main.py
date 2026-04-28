import os
import re
import sys
import traceback
from typing import List, Dict
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import resolve_datasets, get_evaluator_class

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_all_checkpoints(model_path):
    def extract_num(s):
        # Formats like `global_step_397`
        match = re.search(r'^global_step_(\d+)$', s)
        try:
            return int(match.group(1))
        except Exception:
            return -1

    checkpoints = []
    if os.path.isdir(model_path):
        cp_list = [cp for cp in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, cp))]
        if not cp_list or all(extract_num(cp) == -1 for cp in cp_list):
            # use the whole directory as a checkpoint
            checkpoints.append(model_path)
        else:
            cp_list.sort(key=extract_num)
            if len(cp_list) > 1:
                # skip the first epoch
                cp_list = cp_list[1:]
            checkpoints.extend([os.path.join(model_path, item, 'huggingface') for item in cp_list])
    
    print(f"Found {len(checkpoints)} checkpoints")
    return checkpoints


def prepare_save_path(base_dir, dataset, model_path, temperature, n_samples, output_type):
    identifier = model_path.strip("./").replace("/", "--")
    sub_identifier = f"temp_{temperature}_n_{n_samples}"

    save_dir = os.path.join(base_dir, identifier, output_type, sub_identifier, dataset)
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def evaluation(
        base_dir: str,
        ds_mappings: Dict,
        ck_path: str,
        edit_format: str,
        output_type: str,
        use_chat_template: bool,
        parallel_size: int,
        report_k: List[int],
        process_num: int,
        backend: str = "vllm",
        api_model_key: str = None
    ) -> Dict:

    if parallel_size > 0:
        if backend == "vllm":
            from benchmarks.base import vllmGenerator
            generator = vllmGenerator(ck_path, parallel_size)
        elif backend == "api":
            from benchmarks.base import ApiGenerator
            generator = ApiGenerator(api_model_key)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        task = 'generate'
    else:
        generator = None
        task = 'evaluate'

    output_list = [output_type, ]
    
    for output_type in output_list:
        for dataset, param_config in ds_mappings.items():
            evaluator_class = get_evaluator_class(dataset)
            
            # get config
            temperature, n_samples = param_config['temperature'], param_config['n_samples']

            # prepare evaluator
            save_path = prepare_save_path(base_dir, dataset, ck_path, temperature, n_samples, output_type)
            same_args = [generator, {'temperature': temperature, 'n': n_samples}, edit_format, output_type, use_chat_template, report_k, process_num]
            evaluator = evaluator_class(dataset, save_path, *same_args)
            try:
                print(f"Evaluate {dataset} for {ck_path}")
                # main function
                evaluator.main(task if dataset != 'aider' else 'all')
            except Exception:
                print(traceback.format_exc())
    
    if generator and hasattr(generator, "close"):
        generator.close()


if __name__ == '__main__':
    '''
    Same environment for generation
    Require distinct environments for evaluation: humanevalfix, canitedit

    Therefore, we suggest to evaluate via a script
    '''
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets')
    parser.add_argument('--ck_path', type=str, default=None, help="Path to checkpoint (required for vllm backend)")
    parser.add_argument('--report', type=str, default="1")
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--edit_format', type=str, required=True)
    parser.add_argument('--output_type', type=str, default='default')
    parser.add_argument('--parallel_size', type=int, default=4, help="Tensor parallel size for vLLM: 0 for evaluation only")
    parser.add_argument('--process', type=int, default=12, help="Number of processes to use for parallel evaluation")
    parser.add_argument('--backend', type=str, default='vllm', choices=['vllm', 'api'])
    parser.add_argument('--api_model_key', type=str, default=None, help="Model key for API backend")
    parser.add_argument('--temperature', type=float, default=None, help="Override temperature for all selected datasets")
    parser.add_argument('--n_samples', type=int, default=None, help="Override sample count for all selected datasets")
    
    args = parser.parse_args()
    print(args)

    base_dir = args.save_dir
    use_chat_template = args.use_chat_template
    edit_format = args.edit_format
    output_type = args.output_type
    ck_path = args.ck_path
    
    report_k = [int(k.strip()) for k in args.report.split(",")]
    parallel_size = args.parallel_size
    process = args.process
    backend = args.backend
    api_model_key = args.api_model_key

    if backend == 'api':
        if not api_model_key:
            parser.error("--api_model_key is required when --backend api")
        # Use api_model_key as the experiment identifier/save path key.
        ck_path = api_model_key
    else:
        if not ck_path:
            parser.error("--ck_path is required when --backend vllm")

    ds_mappings = resolve_datasets(args.datasets)
    if args.n_samples is not None and args.n_samples <= 0:
        parser.error("--n_samples must be a positive integer")

    if args.temperature is not None or args.n_samples is not None:
        for _, config in ds_mappings.items():
            if args.temperature is not None:
                config['temperature'] = args.temperature
            if args.n_samples is not None:
                config['n_samples'] = args.n_samples

    # evaluation
    if backend == "api":
        checkpoints = [ck_path]
    else:
        checkpoints = find_all_checkpoints(ck_path)

    for one_ck_path in checkpoints:
        evaluation(
            base_dir, ds_mappings, one_ck_path, edit_format, output_type,
            use_chat_template, parallel_size, report_k, process,
            backend=backend, api_model_key=api_model_key
        )
