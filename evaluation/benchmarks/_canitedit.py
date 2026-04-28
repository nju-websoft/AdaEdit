import os
import json
import subprocess
from pathlib import Path

from datasets import load_dataset

from .base import BaseEvaluator


class CanItEditEvaluator(BaseEvaluator):
    def load_dataset(self):
        for item in load_dataset(self.dataset_path, split="test"):
            full_name = item['full_name']

            instr_kind = "instruction_descriptive"
            yield {
                **item,
                "instr_kind": instr_kind,
                "task_id": f"{full_name}_{instr_kind}",
                "instruction": item[instr_kind],
            }

            instr_kind = "instruction_lazy"
            yield {
                **item,
                "instr_kind": instr_kind,
                "task_id": f"{full_name}_{instr_kind}",
                "instruction": item[instr_kind],
            }
    
    def get_starter_code(self):
        ret = {}
        for sample in self.load_dataset():
            ret[sample["task_id"]] = sample["before"]

        return ret
    
    def get_edit_info(self):
        '''
        Note that this function is not required for evaluation.
        '''
        ret = {}
        for sample in self.load_dataset():
            # dataset-specific
            ret[sample['task_id']] = [sample["instruction"], sample["before"], sample["after"]]
        return ret

    def process_prompts(self):
        task_ids = []
        prompts = []
        for sample in self.load_dataset():
            task_ids.append(sample["task_id"])
            prompts.append(self.make_prompt(sample["instruction"], sample["before"]))
        
        return task_ids, prompts

    def process_test_code(self):
        '''
        For analysis
        '''
        task_ids = []     # List[str]
        predictions = []  # List[List[str]]
        
        # load solutions
        with open(self.solution_file, 'r') as f:
            for line in f.readlines():
                item = json.loads(line)
                task_ids.append(item["task_id"])
                predictions.append(item["solutions"])
        
        return task_ids, predictions, []
    
    def prepare_eval_file(self):
        import gzip
        def gunzip_json_write(path, data: dict) -> None:
            with gzip.open(path, "wt") as f:
                json.dump(data, f)
        
        solution_dict = {}
        with open(self.solution_file, 'r') as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                solution_dict[item['task_id']] = item['solutions']
        
        custom_dir = os.path.join(self.save_dir, "outputs")
        os.makedirs(custom_dir, exist_ok=True)
        for sample in self.load_dataset():
            task_id = sample['task_id']
            filename = f"{task_id}.json.gz"
            custom_file = os.path.join(custom_dir, filename)
            if os.path.isfile(custom_file):
                continue

            custom_solution = sample
            custom_solution.update({
                "prompt": "",
                "language": "py",
                "instr_kind": sample['instr_kind'],
                "temperature": self.llm_params['temperature'],
                "completions": solution_dict[task_id]
            })
            gunzip_json_write(custom_file, custom_solution)

        return custom_dir
    
    def evaluate(self) -> dict:
        custom_dir = self.prepare_eval_file()
        # execute evaluation
        cmd = [
            "python", "evaluator_overlay.py", 
            "--dir", custom_dir,
            "--output-dir", custom_dir,
            "--max-workers", str(self.process_num),
        ]
        subprocess.run(cmd, cwd=self.work_dir, check=True)

        # check if the custom_dir has the expected number of result files
        file_count = len(list(Path(custom_dir).glob('*.results.json.gz')))
        if file_count != 210:
            raise ValueError(f"Expected 210 result files, but found {file_count} in {custom_dir}")

        # metrics
        tmp_file = os.path.join(self.save_dir, "tmp_metrics.json")
        cmd = [
            "python", "pass_k.py",
            custom_dir,
            "--report", ','.join([str(k) for k in self.report_k]),
            "--save_file", tmp_file,
        ]
        subprocess.run(cmd, cwd=self.work_dir, check=True)

        # read result
        with open(tmp_file, 'r') as f:
            ret = json.load(f)
        
        assert len(ret) == 1
        info = ret[list(ret.keys())[0]]
        metric_list = [f'pass@{k}' for k in self.report_k]
        metrics = {x: info[x] for x in metric_list if x in info}
        # metrics['excess_code'] = info['excess_code']
        # metrics['excess_code_se'] = info['excess_code_se']
        
        return {self.dataset_name: metrics}


if __name__ == "__main__":
    dataset = "canitedit"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "test-results")
    same_args = (save_dir, None, {'temperature': 0, 'n': 1}, "fullcode")
    
    evaluator = CanItEditEvaluator(dataset, *same_args)
    # use the ground truth as solutions
    with open(os.path.join(save_dir, "raws.jsonl"), 'w') as f:
        for task_id, item in evaluator.get_edit_info().items():
            data = {
                'task_id': task_id,
                'raw_solutions': [item[2],]
            }
            f.write(json.dumps(data) + '\n')

    metrics = evaluator.main()
    print(f"Metrics for {dataset}: {metrics}")