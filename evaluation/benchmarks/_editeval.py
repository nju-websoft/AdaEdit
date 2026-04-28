import os
import json

from .base import BaseEvaluator


class EditEvalEvaluator(BaseEvaluator):
    def load_dataset(self):
        # .jsonl
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
    
    def get_starter_code(self):
        ret = {}
        for sample in self.load_dataset():
            ret[sample['task_id']] = sample["input"]
        return ret
    

    def get_edit_info(self):
        '''
        Note that this function is not required for evaluation.
        '''
        ret = {}
        for sample in self.load_dataset():
            # dataset-specific
            ret[sample['task_id']] = [sample["instruction"], sample["input"], sample["output"]]
        return ret

    def process_prompts(self):
        task_ids = []
        prompts = []
        for sample in self.load_dataset():
            task_ids.append(sample["task_id"])
            prompts.append(self.make_prompt(sample["instruction"], sample["input"]))
        
        return task_ids, prompts
    
    def process_test_code(self):
        task_ids = []     # List[str]
        predictions = []  # List[List[str]]
        references = []   # List[str]

        CODE_MARKER = r"{{Code}}"
        solutions_dict = {}
        with open(self.solution_file, 'r') as f:
            for line in f.readlines():
                item = json.loads(line)
                solutions_dict[item["task_id"]] = item["solutions"]
        
        for item in self.load_dataset():
            task_id = item["task_id"]
            task_ids.append(task_id)
            references.append(item["test"] + "\n\ncheck()")

            context = item.get("context", None)
            solutions = []
            for solution in solutions_dict[task_id]:
                if context and CODE_MARKER in context:
                    solution = context.replace(CODE_MARKER, solution)
                solutions.append(solution)
            predictions.append(solutions)
        
        return task_ids, predictions, references


if __name__ == "__main__":
    dataset = "editeval"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "test-results")
    same_args = (save_dir, None, {'temperature': 0, 'n': 1}, "fullcode")
    
    evaluator = EditEvalEvaluator(dataset, *same_args)
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