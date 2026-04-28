# AdaEdit

To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing, ACL Findings 2026

## Repository Structure

```
AdaEdit/
|-- data_gen/                          # Data preprocessing and SFT data generation
|   |-- preprocess/                    # Dataset filtering and code formatting
|   `-- differ/                        # implementations of diff tools
|-- evaluation/                        # Generation, execution, and reporting scripts
|   |-- benchmarks/                    # Benchmark adapters and dataset path configuration
|   |   |-- config.py                  # Configure external benchmark dataset paths
|   |   `-- testsuits/                 # Local execution helpers, including CanItEdit
|   |-- run.sh                         # vLLM evaluation entry point
|   `-- api_run.sh                     # API-backend evaluation entry point
|-- utils/                             # Shared format parsing and dataset resolution utilities
|-- sft_config.yaml                    # SFT hyperparameters
`-- requirements.txt                   # Python dependencies
```

## Environments

We develop under Ubuntu 22.04.5 LTS.

```
conda create -n AdaEdit python=3.11.13
pip install -r requirements.txt
```

## Prepare Data

Data filtering and code formatting:

```
cd data_gen/preprocess
conda activate AdaEdit

python main.py --save_dir $PRE_DATASET_DIR --dataset_path likaixin/InstructCoder

python main.py --save_dir $PRE_DATASET_DIR --dataset_path zkzhang88/OCEData/ocedata.jsonl --instr_label instruct_purify --input_label code_before_purify --output_label code_after_purify

# we only use the JavaScript subsets of CommitPackFT
python main.py --save_dir $PRE_DATASET_DIR --dataset_path bigcode/commitpackft --instr_label subject --input_label old_contents --output_label new_contents --input_lang_label lang --output_lang_label lang --only_trans
```

Transform training data to different edit formats:

```
cd data_gen
conda activate AdaEdit
python main.py --datasets $PRE_DATASET_DIRS --save_dir $SFT_DATASET_DIR --format $FORMAT
# or: AdaEdit requires specifying a tokenizer to adaptively select edit formats
python main.py --datasets $PRE_DATASET_DIRS --save_dir $SFT_DATASET_DIR --format $FORMAT --model $TOKENIZER_PATH --series_name $SERIES_NAME
```

Example: `... --format ada-blockdiff --model Qwen/Qwen2.5-Coder-7B --series_name Qwen2.5-Coder`

Support edit formats:
- `fullcode`: full-code generation.

- `unidiff` or `unidiff-lineno`: standard unified diff with 3 context lines (without / with line numbers in the source code)
- `minunidiff` or `minunidiff-lineno`: minimal unified diff without context lines (without / with line numbers in the source code)

- `mincontentdiff`: minimal content-addressed diff
- `contentdiff`: content-addressed diff with 3 context lines
- `blockdiff`: our structure-aware diff format
- `funcdiff`: our structure-aware diff format at the function level
- `ada-blockdiff`: our adaptive strategy `AdaEdit` that dynamically chooses the token-efficient format between `blockdiff` and `fullcode`

You can specify different diff shapes for content-addressed diffs:
- the hunk rewrite style: by default
- the unified diff-like style: e.g., `blockdiff-inter`
- the search/replace style: e.g., `blockdiff-search`

## Supervised Fine-Tuning

Use any framework for supervised fine-tuning. See `sft_config.yaml` for hyperparameters.


## Evaluation

See `evaluation/benchmarks/testsuits/README.md` for required benchmark environments.

Before evaluation, prepare the external benchmark datasets and configure their paths in `evaluation/benchmarks/config.py`. Set `ADAEDIT_BENCHMARK_ROOT` to the common benchmark root directory, or edit `DEFAULT_BENCHMARK_ROOT` in `config.py`.

After installation and configuration, you can evaluate a model:

Support output types: `fullcode`, `diff`, `adaptive`

```
cd evaluation
source run.sh $MODEL_PATH $EVAL_SAVE_DIR $EDIT_FORMAT $OUTPUT_TYPE
# or evaluate via API backend
source api_run.sh $API_MODEL_KEY $EVAL_SAVE_DIR $EDIT_FORMAT $OUTPUT_TYPE

# `--datasets` only supports the `edit` preset or explicit dataset names
# (`editeval`, `canitedit`, `aider`, `humanevalfix-python`, `humanevalfix-js`)

conda activate AdaEdit
# accuracy
python accuracy.py --datasets edit --save_dir $FORMAT_EVAL_SAVE_DIR

# patch correctness
python patch_acc.py --datasets edit --save_dir $FORMAT_EVAL_SAVE_DIR --edit_format $FORMAT --output_type $OUTPUT_TYPE

# usability
python usability.py --datasets edit --save_dir $FORMAT_EVAL_SAVE_DIR

# latency and cost
# macro average
python efficiency.py --datasets edit --save_dir $FORMAT_EVAL_SAVE_DIR --model $TOKENIZER_PATH --output_type $OUTPUT_TYPE --macro

# micro average across different lengths of source code, e.g., 300-500 tokens
python efficiency.py --datasets edit --save_dir $FORMAT_EVAL_SAVE_DIR --model $TOKENIZER_PATH --output_type $OUTPUT_TYPE --min_tokens 300 --max_tokens 500

# correctness of adaptive strategy
python adapt_acc.py --datasets edit --save_dir $FORMAT_EVAL_SAVE_DIR --model $TOKENIZER_PATH --edit_format $FORMAT
```

## Citation

If you find the work useful, please kindly cite it as follows:      

```
@inproceedings{AdaEdit,
  author    = {Wei Cheng and Yongchang Cao and Chen Shen and Binhua Li and Jue Chen and Yongbin Li and Wei Hu},
  title     = {To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing},
  booktitle = {ACL},
  year      = {2026}
}
```
