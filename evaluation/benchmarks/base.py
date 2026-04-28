import os
import sys
import json
import re
import time
import random
from abc import ABC, abstractmethod

from .config import DatasetConfig
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import initialize_diff_processor, get_edit_format_prompt_spec


class vllmGenerator:
    def __init__(self, model_path, parallel_size=1):
        from vllm import LLM
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model = LLM(
            model = model_path,
            tokenizer = model_path,
            tensor_parallel_size = parallel_size,   # Total number of attention heads must be divisible by tensor parallel size
            dtype = "bfloat16",
            enable_prefix_caching = True,
            trust_remote_code = True,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.max_length = self.model.llm_engine.model_config.max_model_len

    def close(self):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


class ApiGenerator:
    API_TIMEOUT = 300
    API_TOP_K = 100
    MAX_RETRY = 3
    RETRY_BASE_SLEEP = 1.0
    RETRY_MAX_SLEEP = 8.0
    RETRY_JITTER = 0.3

    def __init__(self, model_key):
        if not model_key:
            raise ValueError("model_key is required for ApiGenerator")

        self.model_key = model_key
        try:
            from request_model import request_model_by_model_key
        except ImportError:
            from evaluation.request_model import request_model_by_model_key

        self.request_model_by_model_key = request_model_by_model_key

    @staticmethod
    def extract_content(response):
        try:
            content = response["output"]["choices"][0]["message"]["content"]
        except Exception:
            return ""
        return content if isinstance(content, str) else ""

    def generate_one(self, messages, temperature, top_p):
        last_error = None
        for attempt in range(self.MAX_RETRY):
            try:
                response = self.request_model_by_model_key(
                    model_key=self.model_key,
                    messages=messages,
                    top_p=top_p,
                    temperature=temperature,
                    top_k=self.API_TOP_K,
                    timeout=self.API_TIMEOUT,
                )
                return self.extract_content(response)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.MAX_RETRY:
                    break

                wait_seconds = min(self.RETRY_MAX_SLEEP, self.RETRY_BASE_SLEEP * (2 ** attempt))
                wait_seconds += random.uniform(0.0, self.RETRY_JITTER)
                time.sleep(wait_seconds)

        print(f"[ApiGenerator] request failed after {self.MAX_RETRY} attempts: {last_error}")
        return ""

    def close(self):
        return


class BaseEvaluator(ABC):
    def __init__(self,
                 dataset_name,
                 save_dir,
                 generator=None,
                 llm_params=None,
                 edit_format="fullcode",
                 output_type=None,
                 use_chat_template=False,
                 report_k=[1,], 
                 process_num=12
                ):
        # get the information of the dataset
        self.dataset_name = dataset_name.lower()
        self.dataset_path = DatasetConfig.get_dataset_path(self.dataset_name)
        self.work_dir = DatasetConfig.get_workdir(self.dataset_name)

        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        self.generator = generator

        # llm hyperparameters
        if llm_params:
            self.report_k = [x for x in report_k if x > 0 and x <= llm_params['n']]
            self.llm_params = {
                'top_p': 1 if llm_params['temperature'] == 0 else 0.95,
                'max_tokens': DatasetConfig.get_maximum_tokens(self.dataset_name),
                'stop': ["\n```"]
            }
            self.llm_params.update(llm_params)
        
        self.global_lang = DatasetConfig.get_languages(self.dataset_name)

        self.edit_format = edit_format
        self.diff_tool, self.output_type, self.enable_lineno = initialize_diff_processor(edit_format, output_type)

        self.use_chat_template = use_chat_template
        self.process_num = process_num

        # Count for multi-turn benchmarks
        self.try_nums = None

    @property
    def raw_file(self) -> str:
        suffix = f"_{self.try_nums}" if self.try_nums is not None else ""
        return os.path.join(self.save_dir, f"raws{suffix}.jsonl")

    @property
    def solution_file(self) -> str:
        suffix = f"_{self.try_nums}" if self.try_nums is not None else ""
        return os.path.join(self.save_dir, f"solutions{suffix}.jsonl")

    @property
    def detail_file(self) -> str:
        suffix = f"_{self.try_nums}" if self.try_nums is not None else ""
        return os.path.join(self.save_dir, f"details{suffix}.json")

    @property
    def metric_file(self) -> str:
        return os.path.join(self.save_dir, "metrics.json")

    @property
    def messages_file(self) -> str:
        suffix = f"_{self.try_nums}" if self.try_nums is not None else ""
        return os.path.join(self.save_dir, f"messages{suffix}.jsonl")


    def _response_prefix_for_output_type(self, tgt_lang):
        if self.output_type == "adaptive":
            return "```"
        if "diff" in self.output_type:
            return "```diff\n"
        if self.output_type == "fullcode":
            return f"```{tgt_lang}\n"
        raise ValueError(f"Invalid output_type: {self.output_type}")


    def make_prompt(self, task, starter_code, src_lang='python', tgt_lang='python'):
        task = task.strip()
        assert task, "Task cannot be empty for making prompts"
        
        if self.enable_lineno and starter_code:
            numbered_lines = [f'{i+1} {line}' for i, line in enumerate(starter_code.splitlines(keepends=True))]
            starter_code = ''.join(numbered_lines)
        
        # ensure ``` at a new line
        if starter_code and not starter_code.endswith('\n'):
            starter_code += '\n'
        
        prompt = f"### Instruction\n{task}\n\n### Input Code\n```{src_lang}\n{starter_code}```\n\n### Response\n"
        
        response_prefix = self._response_prefix_for_output_type(tgt_lang)
        
        if self.use_chat_template and self.generator and hasattr(self.generator, "tokenizer") and hasattr(self.generator.tokenizer, "apply_chat_template"):
            # apply chat template
            prompt_chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_prefix}
            ]
            prompt = self.generator.tokenizer.apply_chat_template(prompt_chat, continue_final_message=True, tokenize=False)
        
        else:
            prompt = prompt + response_prefix
        
        return prompt

    @abstractmethod
    def load_dataset(self):
        '''
        TODO: Override: load dataset
        '''
    
    def get_starter_code(self):
        '''
        TODO: Override this function to get starter code if needed.
        '''
        ret = {} # task_id: starter_code
        for sample in self.load_dataset():
            # dataset-specific
            ret[sample['task_id']] = sample["starter_code"]
        return ret
    
    def get_response_code(self):
        '''
        TODO: Override this function to get response code if needed.
        Note that this function is not required for evaluation.
        '''
        ret = {} # task_id: starter_code
        for sample in self.load_dataset():
            # dataset-specific
            ret[sample['task_id']] = sample["output"]
        return ret
  
    def process_prompts(self):
        '''
        TODO: Override this function to process prompts if needed.
        '''
        task_ids = []
        prompts = []
        for sample in self.load_dataset():
            # dataset-specific
            task_ids.append(sample["task_id"])
            prompts.append(self.make_prompt(sample["instruction"], sample["starter_code"]))
        
        return task_ids, prompts
    
    def process_test_code(self):
        '''
        TODO: Override this function to process test code if needed.
        Final test: prediction + "\n\n" + reference
        '''
        task_ids = []     # List[str]
        predictions = []  # List[List[str]]
        references = []   # List[str]
        
        # load solutions
        solutions_dict = {}
        with open(self.solution_file, 'r') as f:
            for line in f.readlines():
                item = json.loads(line)
                solutions_dict[item["task_id"]] = item["solutions"]

        for sample in self.load_dataset():
            # dataset-specific
            task_id = sample['task_id']
            task_ids.append(task_id)
            references.append(sample['tests'])

            predictions.append(solutions_dict[task_id])
	        
        return task_ids, predictions, references

    def _get_target_language(self):
        if isinstance(self.global_lang, str) and self.global_lang.strip():
            return self.global_lang.strip()
        return "python"

    def _build_api_system_prompt(self, tgt_lang):
        spec = get_edit_format_prompt_spec(self.edit_format, tgt_lang)
        fmt_name = spec["format_name"]
        fmt_desc = spec["description"]
        one_shot = spec["one_shot_example"]

        lines = [
            "You are a coding assistant who edits the input code according to the instruction.\n",
        ]

        if self.output_type == "fullcode":
            lines.append(fmt_desc)
            lines.append(f"Use `{tgt_lang}` as the language tag of the fenced code block.")
            if one_shot:
                lines.append("One-shot format example:")
                lines.append(one_shot)
        elif self.output_type == "diff":
            lines.append(fmt_desc)
            lines.append("Use `diff` as the language tag of the fenced code block.")
            if one_shot:
                lines.append("One-shot format example:")
                lines.append(one_shot)
        elif self.output_type == "adaptive":
            lines.append("Dynamically choose the most token-efficient format between full code and diff format.")
            lines.append(f"If choosing full code, use `{tgt_lang}` as the language tag of the fenced code block.")
            lines.append("If choosing diff format, use `diff` as the language tag of the fenced code block.")
            lines.append('The diff format: ' + fmt_desc)
            if one_shot:
                lines.append("One-shot diff format example:")
                lines.append(one_shot)
        else:
            raise ValueError(f"Invalid output_type: {self.output_type}")

        lines.extend([
            "\nReturn exactly one markdown fenced code block and nothing else.",
            "Do not output explanations outside the code block.",
        ])

        return "\n".join(lines)

    def _strip_api_response_prefix(self, prompt, tgt_lang):
        response_prefix = self._response_prefix_for_output_type(tgt_lang)
        if isinstance(prompt, str) and prompt.endswith(response_prefix):
            return prompt[:-len(response_prefix)]
        return prompt

    def _build_api_messages(self, prompt, tgt_lang):
        user_prompt = self._strip_api_response_prefix(prompt, tgt_lang)
        return [
            {"role": "system", "content": self._build_api_system_prompt(tgt_lang)},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
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

    def _normalize_raw_for_postprocess(self, raw_text):
        if not isinstance(raw_text, str):
            raw_text = "" if raw_text is None else str(raw_text)

        # Keep non-fenced outputs unchanged to preserve existing behavior.
        if "```" not in raw_text:
            if self.output_type == "adaptive" and raw_text == "":
                # avoid splitlines edge case in adaptive parsing
                return "\n"
            return raw_text

        tag, body = self._extract_first_fenced_block(raw_text)
        normalized_body = body.strip("\n") if body else ""

        if self.output_type in {"fullcode", "diff"}:
            return normalized_body if normalized_body else raw_text.strip("\n")

        if self.output_type == "adaptive":
            return f"{tag}\n{normalized_body}"

        raise ValueError(f"Invalid output_type: {self.output_type}")

    def _generate_with_vllm(self, prompts):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            skip_special_tokens=True,
            **self.llm_params
        )
        
        # truncate prompts: mainly use for aider-1
        max_prompt_len = self.generator.max_length - self.llm_params['max_tokens']

        processed_prompt_token_ids = []
        for prompt in prompts:
            token_ids = self.generator.tokenizer.encode(prompt)
            if len(token_ids) > max_prompt_len:
                truncated_token_ids = token_ids[-max_prompt_len:]
                processed_prompt_token_ids.append({"prompt_token_ids": truncated_token_ids})
            else:
                processed_prompt_token_ids.append({"prompt_token_ids": token_ids})

        # code generation
        generations = self.generator.model.generate(prompts=processed_prompt_token_ids, sampling_params=sampling_params)
        return [[completion.text for completion in prompt_generations.outputs] for prompt_generations in generations]

    def _generate_with_api(self, prompts):
        from tqdm import tqdm

        outputs = []
        messages_records = []
        tgt_lang = self._get_target_language()
        sample_num = int(self.llm_params['n'])
        total_requests = len(prompts) * sample_num
        model_key = getattr(self.generator, "model_key", "api")

        pbar = tqdm(
            total=total_requests,
            desc=f"API {self.dataset_name} [{model_key}]",
            unit="req",
            dynamic_ncols=True,
        )

        for prompt in prompts:
            messages = self._build_api_messages(prompt, tgt_lang)
            messages_records.append(messages)
            one_prompt_outputs = []
            for _ in range(sample_num):
                content = self.generator.generate_one(
                    messages=messages,
                    temperature=self.llm_params['temperature'],
                    top_p=self.llm_params['top_p'],
                )
                one_prompt_outputs.append(content if isinstance(content, str) else "")
                pbar.update(1)
            outputs.append(one_prompt_outputs)

        pbar.close()
        return outputs, messages_records

    def generate(self):
        # prepare prompts
        task_ids, prompts = self.process_prompts()
        if not task_ids:
            with open(self.raw_file, "w") as f:
                pass
            if isinstance(self.generator, ApiGenerator):
                with open(self.messages_file, "w") as f:
                    pass
            return

        api_messages = None
        if isinstance(self.generator, ApiGenerator):
            outputs, api_messages = self._generate_with_api(prompts)
        else:
            outputs = self._generate_with_vllm(prompts)
	        
        res = []
        for i, output in enumerate(outputs):
            res.append({
                "task_id": task_ids[i],
                "prompt": prompts[i],
                "raw_solutions": output,
            })

        with open(self.raw_file, "w") as f:
            for item in res:
                f.write(json.dumps(item) + "\n")

        if api_messages is not None:
            with open(self.messages_file, "w") as f:
                for i, messages in enumerate(api_messages):
                    item = {
                        "task_id": task_ids[i],
                        "messages": messages,
                        "llm_params": {
                            "temperature": self.llm_params.get("temperature"),
                            "top_p": self.llm_params.get("top_p"),
                            "n": self.llm_params.get("n"),
                        },
                    }
                    if hasattr(self.generator, "model_key"):
                        item["model_key"] = self.generator.model_key
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Save {len(api_messages)} api messages to {self.messages_file}")
        
        print(f"Save {len(res)} generations to {self.raw_file}")
        
    
    def postprocess(self):
        # Read all lines from input file
        with open(self.raw_file, 'r') as f:
            lines = f.readlines()

        results = []
        if self.output_type == "fullcode":
            # Write results to output file
            for line in lines:
                item = json.loads(line.strip())
                item['solutions'] = [self._normalize_raw_for_postprocess(x) for x in item['raw_solutions']]
                results.append(item)
        else:
            # get starter code
            starter_code_dict = self.get_starter_code()
            if self.output_type == "diff":
                for line in lines:
                    item = json.loads(line.strip())
                    starter_code = starter_code_dict[item['task_id']]
                    solutions = []
                    for x in item['raw_solutions']:
                        x = self._normalize_raw_for_postprocess(x)
                        try:
                            solution = self.diff_tool.apply_diff(starter_code, x)
                        except Exception:
                            solution = starter_code
                        solutions.append(solution)
                    item['solutions'] = solutions
                    results.append(item)
            else:
                for line in lines:
                    item = json.loads(line.strip())
                    starter_code = starter_code_dict[item['task_id']]
                    solutions = []
                    for x in item['raw_solutions']:
                        x = self._normalize_raw_for_postprocess(x)
                        split_lines = x.splitlines(keepends=True)
                        content = x[len(split_lines[0]):] if split_lines else ''
                        if split_lines and split_lines[0].strip() == 'diff':
                            try:
                                solution = self.diff_tool.apply_diff(starter_code, content)
                            except Exception:
                                solution = starter_code
                        else:
                            solution = content
                        solutions.append(solution)
                    item['solutions'] = solutions
                    results.append(item)
        
        # Write results to output file
        with open(self.solution_file, 'w') as fw:
            for result in results:
                fw.write(json.dumps(result) + '\n')
        print(f'Save sanitized generations to {self.solution_file}')

    def execute(self):
        '''
        For benchmarks that seperate execution and evaluation
        '''

    def evaluate(self) -> dict:
        '''
        TODO: Override this function to execute generated code if needed.
        '''
        import numpy as np
        from .testsuits.execution import compute_code_eval, estimate_pass_at_k
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        task_ids, predictions, references = self.process_test_code()
                
        pass_at_k, results = compute_code_eval(
            predictions=predictions,
            references=references,
            k=self.report_k,
            num_workers=self.process_num,
            timeout=3.0
        )
        
        # organize results
        detailed_results = {}
        for idx, task_results in results.items():
            task_id = task_ids[idx]
            passed_list = [r[1]["passed"] for r in task_results]
            result_list = [r[1]["result"] for r in task_results]
            detailed_results[task_id] = [passed_list, result_list]
        
        # pass@k
        task_pass_at_k = {}
        for k_val in self.report_k:
            task_results = {}
            for task_id, (passed_list, _) in detailed_results.items():
                total = len(passed_list)
                if total >= k_val:
                    correct = sum(passed_list)
                    task_results[task_id] = float(
                        estimate_pass_at_k(
                            np.array([total]), 
                            np.array([correct]), 
                            k_val
                        )[0]
                    )
            task_pass_at_k[f"pass@{k_val}"] = task_results
        
        # save results
        output = [{
            **pass_at_k,
            "details": task_pass_at_k
        }, detailed_results]
        
        with open(self.detail_file, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Save execution details to {self.detail_file}")

        metrics = {self.dataset_name: pass_at_k}
        return metrics
    
    def load_metrics(self):
        if os.path.isfile(self.metric_file):
            with open(self.metric_file, 'r') as f:
                return json.load(f)
        return {}

    def save_metrics(self, metrics):
        # load current metrics
        current_metrics = self.load_metrics()
        current_metrics.update(metrics)

        self.print_metrics(current_metrics)
        with open(self.metric_file, 'w') as f:
            json.dump(current_metrics, f, indent=4)
        print(f"Save metrics to {self.metric_file}")

        return current_metrics
    
    def print_metrics(self, metrics):
        for dataset, values in metrics.items():
            print(f"# {dataset}")
            for key, value in values.items():
                if isinstance(value, int):
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {100*value:.2f}")
    
    def is_valid_file(self, filename, valid_key):
        if os.path.isfile(filename) and os.stat(filename).st_size > 0:
            with open(filename, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if valid_key not in item:
                        return False
            return True

        return False
    
    def check_enough_metrics(self) -> bool:
        metrics = self.load_metrics()
        current_ds = self.dataset_name
        if self.try_nums is not None:
            current_ds += f'_{self.try_nums}'

        if current_ds not in metrics:
            return False

        return True
    
    def main(self, mode="all"):
        '''
        mode: all, generate, execute, evaluate
        '''
        if mode in ["all", "generate"]:
            if not self.is_valid_file(self.raw_file, 'raw_solutions'):
                self.generate()

            if not self.is_valid_file(self.solution_file, 'solutions'):
                self.postprocess()
        
        is_completed = self.check_enough_metrics()
        if mode in ["all", "execute"] and not is_completed:
            self.execute()
        
        if mode in ["all", "evaluate"]:
            if not is_completed:
                metrics = self.evaluate()
                metrics = self.save_metrics(metrics)
            else:
                metrics = self.load_metrics()
            
            return metrics
