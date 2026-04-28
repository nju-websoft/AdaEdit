'''
Preprocess the dataset to parquet format
'''
import os
import sys
import time

from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import initialize_diff_processor


class SFTDatasetGenerator:
    def __init__(self, edit_format, model_path=None, use_chat_template=False):
        self.edit_format = edit_format
        self.enable_lineno = False
        self.diff_tool, self.output_type, self.enable_lineno = initialize_diff_processor(edit_format)
        
        if model_path:
            # load tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.use_chat_template = use_chat_template
        else:
            self.use_chat_template = False
        
        self.diff_times = None
        self.patch_times = None


    def make_prompt(self, task, starter_code, language):
        task = task.strip()
        assert task, "Task cannot be empty for making prompts"
        
        if self.enable_lineno and starter_code:
            numbered_lines = [f'{i+1} {line}' for i, line in enumerate(starter_code.splitlines(keepends=True))]
            starter_code = ''.join(numbered_lines)
        
        # ensure ``` at a new line
        if starter_code and not starter_code.endswith('\n'):
            starter_code += '\n'
        
        prompt = f"### Instruction\n{task}\n\n### Input Code\n```{language}\n{starter_code}```\n\n### Response\n"
        
        return prompt
    

    def if_use_diff(self, text, diff):
        '''Only use diff when the diff is shorter than the text'''
        return len(self.tokenizer.encode(diff, add_special_tokens=False)) < len(self.tokenizer.encode(text, add_special_tokens=False))


    def handle_func(self, instance):
        code_before = instance['before']
        code_after = instance['after']
        lang_before = instance['before_lang']
        lang_after = instance['after_lang']

        prompt = self.make_prompt(instance['task'], code_before, lang_before)

        if self.use_chat_template:
            # apply chat template
            prompt_chat = [
                {"role": "user", "content": prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        
        response_prefix = None
        if self.output_type in ['diff', 'adaptive']:
            stime = time.time()
            response = self.diff_tool.calculate_diff(code_before, code_after, lang=lang_before)
            # specify diff or selecting diff when adaptive
            if self.output_type == 'diff' or self.if_use_diff(code_after, response):
                response_prefix = '```diff\n'
            self.diff_times.append(time.time() - stime)
        
        if not response_prefix:
            response = code_after
            response_prefix = f'```{lang_after}\n'
        
        if response and not response.endswith('\n'):
            response += '\n'

        return {
            'prompt': prompt,
            'input_code': code_before,
            'response': response_prefix + response + '```'
        }


    def convert_dataset(self, dataset_path, save_dir, series_name):
        dataset = load_from_disk(dataset_path)
        ds_name = os.path.basename(dataset_path)

        print(f'{len(dataset)} instances in {ds_name}')

        # initialize
        self.diff_times = []
        self.patch_times = []
        df = []
        diff_count = 0
        for instance in tqdm(dataset, desc="Process dataset", mininterval=5.0):
            item_info = self.handle_func(instance)
            if not item_info:
                continue

            if not isinstance(item_info, list):
                item_info = [item_info]

            for item in item_info:
                if item['response'].startswith('```diff'):
                    diff_count += 1

                df.append({'prompt': item['prompt'], 'input_code': item['input_code'], 'response': item['response']})
        print(f"Diff count: {diff_count} / {len(df)}")

        if self.diff_times:
            print('Times:')
            print(f"  Avg diff time: {sum(self.diff_times) / len(self.diff_times)}")
            print(f"  Max diff time: {max(self.diff_times)}")
            print(f"  Avg patch time: {sum(self.patch_times) / len(self.patch_times)}")
            print(f"  Max patch time: {max(self.patch_times)}")

        if series_name:
            prompt_type = 'Instruct' if self.use_chat_template else 'Base'
            save_path = os.path.join(save_dir, self.edit_format, series_name, prompt_type, ds_name)
        else:
            save_path = os.path.join(save_dir, self.edit_format, ds_name)

        os.makedirs(save_path, exist_ok=True)
        pd.DataFrame(df).to_parquet(os.path.join(save_path, 'all.parquet'))
        print(f"Saved all sets with {len(df)} rows to {save_path}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+')
    parser.add_argument('--format', type=str, required=True)

    # for using chat template for instruct models or requiring tokenizer
    parser.add_argument('--model', type=str)
    parser.add_argument('--series_name', type=str)   # a unique label for the model series
    parser.add_argument('--use_chat_template', action='store_true')
    
    args = parser.parse_args()

    edit_format = args.format
    model_path = args.model
    series_name = args.series_name
    use_chat_template = args.use_chat_template
    if 'ada' in edit_format or use_chat_template:
        # require tokenizer
        assert model_path and series_name
    else:
        model_path = None
        series_name = None

    generator = SFTDatasetGenerator(edit_format, model_path, use_chat_template)
    for dataset_path in args.datasets:
        generator.convert_dataset(dataset_path, args.save_dir, series_name)
