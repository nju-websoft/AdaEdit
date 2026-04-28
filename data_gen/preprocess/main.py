import os

from tqdm import tqdm
from datasets import load_dataset, Dataset


DEFAULT_LANGUAGE = 'python'
SUPPORTED_LANGUAGES = {'python'}

def process_dataset(dataset: Dataset, label_dict: dict):
    '''
    Process the dataset by removing syntax errors (only check code after), 
    removing semantic unchanged pairs, and finally formatting the code.
    '''
    from python_parser import PythonParser, syntax_check
    parser = PythonParser()

    ds = []
    for sample in tqdm(dataset, desc="Processing"):
        output_lang = sample.get(label_dict['output_lang'], DEFAULT_LANGUAGE).lower()
        assert output_lang in SUPPORTED_LANGUAGES, f"Unsupport processing samples for {output_lang}"

        input_lang = sample.get(label_dict['input_lang'], DEFAULT_LANGUAGE).lower()

        raw_before = sample[label_dict['input']]
        raw_after = sample[label_dict['output']]

        if not raw_before.strip() or not raw_after.strip():
            continue

        # remove pairs with syntax errors in the after code
        if not syntax_check(raw_after):
            continue

        # remove pairs without semantic changes
        before = parser.normalize_code(raw_before)
        after = parser.normalize_code(raw_after)
        if before == after:
            continue
        
        # format code
        before = parser.format_code(raw_before)
        after = parser.format_code(raw_after)
        ds.append({
            'task': sample[label_dict['instruction']],
            'before': before,
            'after': after,
            'before_lang': input_lang,
            'after_lang': output_lang,
        })
    
    return ds


def transform_dataset(dataset: Dataset, label_dict: dict):
    ds = dataset.map(lambda sample: {
        'task': sample[label_dict['instruction']],
        'before': sample[label_dict['input']],
        'after': sample[label_dict['output']],
        'before_lang': sample.get(label_dict['input_lang'], DEFAULT_LANGUAGE).lower(),
        'after_lang': sample.get(label_dict['output_lang'], DEFAULT_LANGUAGE).lower(),
    })

    ds = ds.filter(lambda x: x['before'].strip() and x['after'].strip() and x['before'].strip() != x['after'].strip())
    return ds


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--instr_label', type=str, default='instruction')
    parser.add_argument('--input_label', type=str, default='input')
    parser.add_argument('--output_label', type=str, default='output')
    parser.add_argument('--input_lang_label', type=str, default='python')
    parser.add_argument('--output_lang_label', type=str, default='python')
    parser.add_argument('--only_trans', action='store_true')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if os.path.isfile(dataset_path):
        ds_name = os.path.basename(os.path.dirname(dataset_path))
    else:
        ds_name = os.path.basename(dataset_path)
    save_path = os.path.join(save_dir, ds_name)

    # load the dataset
    if os.path.isfile(dataset_path) and dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")
    
    print(f"Loading dataset from {dataset_path}: {len(dataset)} samples")

    label_dict = {
        "instruction": args.instr_label,
        "input": args.input_label,
        "output": args.output_label,
        "input_lang": args.input_lang_label,
        "output_lang": args.output_lang_label,
    }

    if args.only_trans:
        handled_dataset = transform_dataset(dataset, label_dict)
    else:
        handled_dataset = process_dataset(dataset, label_dict)

    # save the handled dataset
    if not isinstance(handled_dataset, Dataset):
        handled_dataset = Dataset.from_list(list(handled_dataset))
    
    handled_dataset = handled_dataset.add_column("index", list(range(len(handled_dataset))))
    print(f"Processed dataset with {len(handled_dataset)} samples.")
    handled_dataset.save_to_disk(save_path)
