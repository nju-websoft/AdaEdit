import os
import json
# import subprocess

from datasets import load_dataset

from .base import BaseEvaluator


class HumanEvalFixEvaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_num = 4
        self.lang_label = 'js' if self.global_lang == 'javascript' else self.global_lang

    def load_dataset(self):
        for item in load_dataset(self.dataset_path, self.lang_label)["test"]:
            yield item
    
    def get_starter_code(self):
        ret = {}
        for sample in self.load_dataset():
            ret[sample['task_id']] = (sample["declaration"] + sample["buggy_solution"]).strip()
        return ret

    
    def get_edit_info(self):
        '''
        Note that this function is not required for evaluation.
        '''
        ret = {}
        for sample in self.load_dataset():
            # dataset-specific
            task = f'Based on the correct tests, fix bugs in `{sample["entry_point"]}`.\n\n'
            task += f'```{self.global_lang}\n' + sample["test"].strip() + '\n```'

            starter_code = sample["declaration"] + sample["buggy_solution"]
            reference = sample["declaration"] + sample["canonical_solution"]
            ret[sample['task_id']] = [task, starter_code.strip(), reference.strip()]
        return ret

    def process_prompts(self):
        task_ids = []
        prompts = []

        starter_code_dict = self.get_starter_code()
        for sample in self.load_dataset():
            task_id = sample["task_id"]
            task_ids.append(task_id)

            task = f'Based on the correct tests, fix bugs in `{sample["entry_point"]}`.\n\n'
            task += f'```{self.global_lang}\n' + sample["test"].strip() + '\n```'
            prompts.append(self.make_prompt(task, starter_code_dict[task_id], self.global_lang, self.global_lang))
        
        return task_ids, prompts
    
    def process_test_code(self):
        def remove_test_block(solution: str, test: str) -> str:
            test_lines = [line.strip() for line in test.splitlines(keepends=True) if line.strip()]
            if not test_lines:
                return solution
            anchor_lines = ["".join(s.split()) for s in [test_lines[0], test_lines[-1]]]

            solution_lines = solution.splitlines(keepends=True)
            for i, line in enumerate(solution_lines):
                if "".join(line.split()) in anchor_lines:
                    return ''.join(solution_lines[:i])
            return solution
        
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
            references.append(sample['test'])

            solutions = [remove_test_block(x, sample["test"]) for x in solutions_dict[task_id]]
            predictions.append(solutions)
        
        return task_ids, predictions, references
    

    def evaluate(self) -> dict:
        import numpy as np
        from .testsuits.execution import compute_code_eval, estimate_pass_at_k
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        task_ids, predictions, references = self.process_test_code()
                
        pass_at_k, results = compute_code_eval(
            predictions=predictions,
            references=references,
            k=self.report_k,
            num_workers=self.process_num,
            timeout=5.0,
            language=self.global_lang
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


    # def prepare_eval_file(self):
    #     custom_file = os.path.join(self.save_dir, "custom_solutions.json")
        
    #     solution_dict = {}
    #     with open(self.solution_file, 'r') as f:
    #         for line in f.readlines():
    #             item = json.loads(line.strip())
    #             id_num = int(item['task_id'].split('/')[-1])
    #             solution_dict[id_num] = item['solutions']

    #     # [[], ]
    #     custom_solutions = [item[1] for item in sorted(solution_dict.items())]

    #     with open(custom_file, 'w') as f:
    #         json.dump(custom_solutions, f, indent=4)

    #     return custom_file

    # def evaluate(self) -> dict:
    #     script_path = os.path.join(self.work_dir, "main.py")
    #     task = f"humanevalfixtests-{self.lang_label}"
    #     custom_file = self.prepare_eval_file()
    #     tmp_file = os.path.join(self.save_dir, "tmp_metrics.json")
    #     cmd = [
    #         "python",
    #         script_path,
    #         "--tasks", task,
    #         "--n_samples", str(self.llm_params['n']),
    #         "--load_generations_path", custom_file,
    #         "--metric_output_path", tmp_file,
    #         "--allow_code_execution"
    #     ]
    #     subprocess.run(cmd, cwd=self.save_dir, check=True)

    #     with open(tmp_file, 'r') as f:
    #         details = json.load(f)
        
    #     target_k = {f'pass@{k}' for k in self.report_k}
    #     metrics = {
    #         self.dataset_name: {
    #             k: v for k, v in details[task].items() if k in target_k
    #         }
    #     }

    #     return metrics


if __name__ == "__main__":
    # You may install js-md5 module to reach the perfect score.
    dataset = "humanevalfix-js"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "test-results")
    same_args = (save_dir, None, {'temperature': 0, 'n': 1}, "fullcode")
    
    evaluator = HumanEvalFixEvaluator(dataset, *same_args)

    with open(os.path.join(save_dir, "raws.jsonl"), 'w') as f:
        for task_id, item in evaluator.get_edit_info().items():
            data = {
                'task_id': task_id,
                'raw_solutions': [item[2],]
            }
            f.write(json.dumps(data) + '\n')

    metrics = evaluator.main()
    print(f"Metrics for {dataset}: {metrics}")
