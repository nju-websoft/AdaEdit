import os


DEFAULT_BENCHMARK_ROOT = "/path/to/benchmarks"
BENCHMARK_ROOT = os.environ.get("ADAEDIT_BENCHMARK_ROOT", DEFAULT_BENCHMARK_ROOT)


class DatasetConfig:
    '''
    TODO You should prepare benchmarks and modify DEFAULT_BENCHMARK_ROOT
    or set ADAEDIT_BENCHMARK_ROOT.
    '''
    # Benchmarks

    # wget: https://github.com/qishenghu/InstructCoder/raw/refs/heads/main/edit_eval/edit_eval.jsonl
    EDITEVAL_DATASET = os.path.join(BENCHMARK_ROOT, "edit_eval.jsonl")

    # HuggingFace
    CANITEDIT_DATASET = os.path.join(BENCHMARK_ROOT, "CanItEdit")

    # `testsuits/CanItEdit`, which is from https://github.com/nuprl/CanItEdit and https://github.com/nuprl/MultiPL-E
    CANITEDIT_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testsuits", "CanItEdit")

    # HuggingFace
    HUMANEVALFIX_DATASET = os.path.join(BENCHMARK_ROOT, "humanevalpack")

    # git clone: https://github.com/bigcode-project/bigcode-evaluation-harness
    HUMANEAVLFIX_CODE = os.path.join(BENCHMARK_ROOT, "source_code", "bigcode-evaluation-harness")

    # git clone: https://github.com/exercism/python/tree/main/exercises/practice
    AIDER_DATASET = os.path.join(BENCHMARK_ROOT, "python", "exercises", "practice")

    @staticmethod
    def get_dataset_path(dataset):
        if dataset == "editeval":
            return DatasetConfig.EDITEVAL_DATASET
        elif dataset == "canitedit":
            return DatasetConfig.CANITEDIT_DATASET
        elif dataset in {"humanevalfix-python", 'humanevalfix-js'}:
            return DatasetConfig.HUMANEVALFIX_DATASET
        elif dataset == "aider":
            return DatasetConfig.AIDER_DATASET
        else:
            raise ValueError(f"Dataset {dataset} not supported")
    
    @staticmethod
    def get_workdir(dataset):
        if dataset == "canitedit":
            return DatasetConfig.CANITEDIT_CODE
        elif "humanevalfix" in dataset:
            return DatasetConfig.HUMANEAVLFIX_CODE
        
        return None
    

    @staticmethod
    def get_maximum_tokens(dataset):
        return 4096


    @staticmethod
    def get_languages(dataset):
        if dataset in {"humanevalfix-python", "editeval", "canitedit", "aider"}:
            return "python"
        elif dataset == "humanevalfix-js":
            return "javascript"
        
        # decide on each sample
        return None
