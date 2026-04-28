import os
import json

from .base import BaseEvaluator


class AiderEvaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.try_nums = 0
        self.MAX_RETRY = 2
    

    def initialize(self, jsonl_file):
        from pathlib import Path

        ds = []
        for subdir_path in Path(self.dataset_path).iterdir():
            if subdir_path.is_dir():
                task_id = subdir_path.name

                config_path = subdir_path / ".meta" / "config.json"
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                file_info = config["files"]
                assert len(file_info["solution"]) == 1, task_id
                assert len(file_info["example"]), task_id

                file_info["solution"] = os.path.basename(file_info["solution"][0])
                file_info["example"] = file_info["example"][0]
                starter_code = (subdir_path / file_info["solution"]).read_text()
                example_code = (subdir_path / file_info["example"]).read_text()

                for i, test_file in enumerate(file_info["test"]):
                    file_info["test"][i] = (os.path.basename(test_file), (subdir_path / test_file).read_text())
                
                util_filename = "test_utils.py"
                util_filepath = subdir_path / util_filename
                if util_filepath.exists():
                    for test_file in file_info["test"]:
                        if util_filename == test_file[0]:
                            break
                    else:
                        file_info["test"].append((util_filename, (subdir_path / util_filename).read_text()))
                
                instructions = ""
                introduction = subdir_path / ".docs/introduction.md"
                if introduction.exists():
                    instructions += introduction.read_text()
                instructions += (subdir_path / ".docs/instructions.md").read_text()
                instructions_append = subdir_path / ".docs/instructions.append.md"
                if instructions_append.exists():
                    instructions += instructions_append.read_text()

                ds.append({
                    "task_id": task_id,
                    "instruction": instructions,
                    "starter_code": starter_code,
                    "canonical_solution": example_code,
                    "files": file_info
                })
        
        with open(jsonl_file, 'w') as f:
            for data in ds:
                f.write(json.dumps(data) + '\n')


    def load_dataset(self):
        jsonl_file = os.path.join(self.dataset_path, 'aider.jsonl')

        # first use: transform dataset file
        if not os.path.isfile(jsonl_file):
            self.initialize(jsonl_file)

        # .jsonl
        with open(jsonl_file, 'r') as f:
            for line in f:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
    

    def get_starter_code(self):
        ret = {} # task_id: starter_code

        if self.try_nums == 0:
            for sample in self.load_dataset():
                # dataset-specific
                ret[sample['task_id']] = sample["starter_code"]
        else:
            # load solutions
            last_solution_file = os.path.join(self.save_dir, f"solutions_{self.try_nums-1}.jsonl")
            with open(last_solution_file, 'r') as f:
                for line in f.readlines():
                    item = json.loads(line)
                    ret[item["task_id"]] = item["solutions"][0]

        return ret
    

    def get_edit_info(self):
        '''
        Note that this function is not required for evaluation.
        '''
        ret = {}
        for sample in self.load_dataset():
            # dataset-specific
            ret[sample['task_id']] = [sample["instruction"], sample["starter_code"], sample["canonical_solution"]]
        return ret


    def process_prompts(self):
        task_ids = []
        prompts = []

        if self.try_nums == 0:
            for sample in self.load_dataset():
                # dataset-specific
                task_ids.append(sample["task_id"])
                prompts.append(self.make_prompt(sample["instruction"], sample["starter_code"]))
        else:
            # load solutions
            starter_code_dict = self.get_starter_code()

            last_detail_file = os.path.join(self.save_dir, f"details_{self.try_nums-1}.json")
            with open(last_detail_file, 'r') as f:
                last_details = json.load(f)[1]

            for task_id, (passed_list, result_list) in last_details.items():
                if not passed_list[0]:
                    # Not passed
                    task_ids.append(task_id)
                    instruction = 'Fix the code to resolve the testing errors.\n\n'

                    # only first 50 lines: refer to Aider
                    errors = ''.join(result_list[0].splitlines(keepends=True)[:50])
                    instruction += '```text\n' + errors.strip() + '\n```'
                    prompts.append(self.make_prompt(instruction, starter_code_dict[task_id]))
            
        return task_ids, prompts

    
    def process_test_code(self):
        task_ids = []     # List[str]
        predictions = []  # List[List[str]]
        references = []
        
        # load solutions
        solutions_dict = {}
        with open(self.solution_file, 'r') as f:
            for line in f.readlines():
                item = json.loads(line)
                solutions_dict[item["task_id"]] = item["solutions"]

        for sample in self.load_dataset():
            # dataset-specific
            task_id = sample['task_id']
            if task_id in solutions_dict:
                task_ids.append(task_id)
                references.append(sample['files'])
                predictions.append(solutions_dict[task_id])
        
        return task_ids, predictions, references


    def evaluate(self) -> dict:
        import numpy as np
        from .testsuits.execution import compute_code_eval_pytest, estimate_pass_at_k
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        task_ids, predictions, references = self.process_test_code()

        detailed_results = {}
        
        if task_ids:
            pass_at_k, results = compute_code_eval_pytest(
                predictions=predictions,
                file_infos=references,
                k=self.report_k,
                num_workers=self.process_num,
                timeout=180
            )
            
            # organize results
            for idx, task_results in results.items():
                task_id = task_ids[idx]
                passed_list = [r[1]["passed"] for r in task_results]
                result_list = [r[1]["result"] for r in task_results]
                detailed_results[task_id] = [passed_list, result_list]
        
        if self.try_nums > 0:
            # add previous results
            for sample in self.load_dataset():
                task_id = sample["task_id"]
                if task_id not in detailed_results:
                    detailed_results[task_id] = [[True, ], None]
        
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
        
        if self.try_nums > 0:
            # re-calculate pass_at_k
            pass_at_k = {k: sum(v.values()) / len(v) for k, v in task_pass_at_k.items()}
        
        # save results
        output = [{
            **pass_at_k,
            "details": task_pass_at_k
        }, detailed_results]
        
        with open(self.detail_file, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Save execution details to {self.detail_file}")

        metrics = {f'{self.dataset_name}_{self.try_nums}': pass_at_k}

        return metrics

    
    def main(self, mode="all"):
        assert self.llm_params['n'] == 1, "Only support n_samples=1 for Aider"
        self.try_nums = 0

        # generate - evaluate - generate - evaluate
        metrics = None
        for _ in range(self.MAX_RETRY):
            if not self.is_valid_file(self.raw_file, 'raw_solutions'):
                self.generate()

            if not self.is_valid_file(self.solution_file, 'solutions'):
                self.postprocess()
            
            is_completed = self.check_enough_metrics()
        
            if not is_completed:
                metrics = self.evaluate()
                metrics = self.save_metrics(metrics)
            else:
                metrics = self.load_metrics()

            self.try_nums += 1
            
        return metrics


if __name__ == "__main__":
    dataset = "aider"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "test-results")
    same_args = (save_dir, None, {'temperature': 0, 'n': 1}, "fullcode")
    
    evaluator = AiderEvaluator(dataset, *same_args)
    with open(os.path.join(save_dir, "raws_0.jsonl"), 'w') as f:
        for task_id, item in evaluator.get_edit_info().items():
            data = {
                'task_id': task_id,
                'raw_solutions': [item[2],]
            }
            f.write(json.dumps(data) + '\n')

    metrics = evaluator.main()
    print(f"Metrics for {dataset}: {metrics}")