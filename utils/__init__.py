import os
import sys


def _parse_edit_format(edit_format: str) -> dict:
    fmt = (edit_format or "").strip().lower()
    is_adaptive = False
    enable_lineno = False
    diff_shape = "standard"
    prompt_shape = "standard"

    if fmt.startswith("ada-"):
        is_adaptive = True
        fmt = fmt[4:]

    if fmt.endswith("-lineno"):
        enable_lineno = True
        fmt = fmt[:-7]

    if fmt.endswith("-inter"):
        diff_shape = "interlaced"
        prompt_shape = "inter"
        fmt = fmt[:-6]
    elif fmt.endswith("-search"):
        diff_shape = "search_replace"
        prompt_shape = "search"
        fmt = fmt[:-7]

    return {
        "base_format": fmt,
        "is_adaptive": is_adaptive,
        "enable_lineno": enable_lineno,
        "diff_shape": diff_shape,
        "prompt_shape": prompt_shape,
    }


def _normalize_edit_format_for_prompt(edit_format: str) -> str:
    return _parse_edit_format(edit_format)["base_format"]


def get_edit_format_prompt_spec(edit_format: str, tgt_lang: str) -> dict:
    """
    Return prompt-facing spec for an edit format.
    Note: `-lineno` is ignored because it only changes input rendering.
    """
    parsed_format = _parse_edit_format(edit_format)
    fmt = parsed_format["base_format"]

    if fmt == "fullcode":
        return {
            "format_name": "fullcode",
            "description": "Return full source code.",
            "one_shot_example": f'''
```{tgt_lang}
def process_user_batch(users):
    for user in users:
        if not check_active(user):
            print("Warning.")
            process_user(user, False)
            continue

        if user.age >= 16:
            process_user(user)
```'''.strip(),
        }

    if fmt == "minunidiff":
        return {
            "format_name": "minunidiff",
            "description": "Return the unified diff without context lines (-U0).",
            "one_shot_example": '''
```diff
@@ -5 +5 @@
-            process_user(user)
+            process_user(user, False)
```'''.strip(),
        }

    if fmt == "unidiff":
        return {
            "format_name": "unidiff",
            "description": "Return the unified diff with three context lines (-U3).",
            "one_shot_example": '''
```diff
@@ -2,7 +2,7 @@
     for user in users:
         if not check_active(user):
             print("Warning.")
-            process_user(user)
+            process_user(user, False)
             continue
 
         if user.age >= 16:
```'''.strip(),
        }

    shape = parsed_format["prompt_shape"]

    if fmt == "mincontentdiff":
        description = "Return the content-addressed diff without line numbers, which identifies each edit region by a minimal but unique anchor content."
        if shape == "standard":
            description += "\nAdopt the hunk rewrite style, which directly specifies the target code block to be rewritten."
            one_shot_example = '''
```diff
@@ ... @@
-            print("Warning.")
-            process_user(user)
-            continue
+            print("Warning.")
+            process_user(user, False)
+            continue
```'''.strip()
        elif shape == "inter":
            description += "\nAdopt the unified diff-like style, which incorporates context lines alongside addition and deletion markers."
            one_shot_example = '''
```diff
@@ ... @@
             print("Warning.")
-            process_user(user)
+            process_user(user, False)
             continue
```'''.strip()
        elif shape == "search":
            description += "\nAdopt the search/replace style, which utilizes specific delimiters to separate the original block from its replacement."
            one_shot_example = '''
```diff
@@ ... @@
<<<<<<< SEARCH
            print("Warning.")
            process_user(user)
            continue
=======
            print("Warning.")
            process_user(user, False)
            continue
>>>>>>> REPLACE
```'''.strip()
        
        return {
            "format_name": "mincontentdiff",
            "description": description,
            "one_shot_example": one_shot_example,
        }

    if fmt == "contentdiff":
        description = "Return the content-addressed diff without line numbers, which identifies each edit region by a unique anchor content and requires at least three context lines."
        if shape == "standard":
            description += "\nAdopt the hunk rewrite style, which directly specifies the target code block to be rewritten."
            one_shot_example = '''
```diff
@@ ... @@
-    for user in users:
-        if not check_active(user):
-            print("Warning.")
-            process_user(user)
-            continue
-
-        if user.age >= 16:
+    for user in users:
+        if not check_active(user):
+            print("Warning.")
+            process_user(user, False)
+            continue
+
+        if user.age >= 16:
```'''.strip()
        elif shape == "inter":
            description += "\nAdopt the unified diff-like style, which incorporates context lines alongside addition and deletion markers."
            one_shot_example = '''
```diff
@@ ... @@
     for user in users:
         if not check_active(user):
             print("Warning.")
-            process_user(user)
+            process_user(user, False)
             continue
 
         if user.age >= 16:
```'''.strip()
        elif shape == "search":
            description += "\nAdopt the search/replace style, which utilizes specific delimiters to separate the original block from its replacement."
            one_shot_example = '''
```diff
@@ ... @@
<<<<<<< SEARCH
    for user in users:
        if not check_active(user):
            print("Warning.")
            process_user(user)
            continue

        if user.age >= 16:
=======
    for user in users:
        if not check_active(user):
            print("Warning.")
            process_user(user, False)
            continue

        if user.age >= 16:
>>>>>>> REPLACE
```'''.strip()
        
        return {
            "format_name": "contentdiff",
            "description": description,
            "one_shot_example": one_shot_example,
        }

    if fmt == "blockdiff":
        description = "Return the content-addressed diff without line numbers, which represent changes as block-level rewrites of code control structures, e.g., branches, loops, contextual blocks, and functions."
        if shape == "standard":
            description += "\nAdopt the hunk rewrite style, which directly specifies the target code block to be rewritten."
            one_shot_example = '''
```diff
@@ ... @@
-        if not check_active(user):
-            print("Warning.")
-            process_user(user)
-            continue
+        if not check_active(user):
+            print("Warning.")
+            process_user(user, False)
+            continue
```'''.strip()
        elif shape == "inter":
            description += "\nAdopt the unified diff-like style, which incorporates context lines alongside addition and deletion markers."
            one_shot_example = '''
```diff
@@ ... @@
         if not check_active(user):
             print("Warning.")
-            process_user(user)
+            process_user(user, False)
             continue
```'''.strip()
        elif shape == "search":
            description += "\nAdopt the search/replace style, which utilizes specific delimiters to separate the original block from its replacement."
            one_shot_example = '''
```diff
@@ ... @@
<<<<<<< SEARCH
        if not check_active(user):
            print("Warning.")
            process_user(user)
            continue
=======
        if not check_active(user):
            print("Warning.")
            process_user(user, False)
            continue
>>>>>>> REPLACE
```'''.strip()
        
        return {
            "format_name": "blockdiff",
            "description": description,
            "one_shot_example": one_shot_example,
        }

    if fmt == "funcdiff":
        description = "Return the content-addressed diff without line numbers, which represent changes as block-level rewrites of code functions."
        if shape == "standard":
            description += "\nAdopt the hunk rewrite style, which directly specifies the target code block to be rewritten."
            one_shot_example = '''
```diff
@@ ... @@
-def process_user_batch(users):
-    for user in users:
-        if not check_active(user):
-            print("Warning.")
-            process_user(user)
-            continue
-
-        if user.age >= 16:
-            process_user(user)
+def process_user_batch(users):
+    for user in users:
+        if not check_active(user):
+            print("Warning.")
+            process_user(user, False)
+            continue
+
+        if user.age >= 16:
+            process_user(user)
```'''.strip()
        elif shape == "inter":
            description += "\nAdopt the unified diff-like style, which incorporates context lines alongside addition and deletion markers."
            one_shot_example = '''
```diff
@@ ... @@
 def process_user_batch(users):
     for user in users:
         if not check_active(user):
             print("Warning.")
-            process_user(user)
+            process_user(user, False)
             continue
 
         if user.age >= 16:
             process_user(user)
```'''.strip()
        elif shape == "search":
            description += "\nAdopt the search/replace style, which utilizes specific delimiters to separate the original block from its replacement."
            one_shot_example = '''
```diff
@@ ... @@
<<<<<<< SEARCH
def process_user_batch(users):
    for user in users:
        if not check_active(user):
            print("Warning.")
            process_user(user)
            continue

        if user.age >= 16:
            process_user(user)
=======
def process_user_batch(users):
    for user in users:
        if not check_active(user):
            print("Warning.")
            process_user(user, False)
            continue

        if user.age >= 16:
            process_user(user)
>>>>>>> REPLACE
```'''.strip()
        
        return {
            "format_name": "funcdiff",
            "description": description,
            "one_shot_example": one_shot_example,
        }

    raise ValueError(f"Unsupported edit format for prompting: {edit_format}")


def initialize_diff_processor(edit_format, specify_type=None):
    """
    Initialize diff processor based on output format.

    All supported edit formats:
    - fullcode
    - number-indexed diff formats:
        - unidiff
        - minunidiff
        - unidiff-lineno
        - minunidiff-lineno
    - content-addressed diff formats:
        - mincontentdiff
        - contentdiff
        - blockdiff
        - funcdiff
    
    For content-addressed diff formats, you can specify different diff shapes:
    - standard (default)
    - interlaced, e.g., blockdiff-inter
    - search_replace, e.g., blockdiff-search

    Another option for diff formats is adaptive strategy, e.g., ada-blockdiff
    
    Args:
        edit_format (str): The trained edit format
        specify_type (str | None): The specified output type (fullcode, diff, adaptive)
        
    Returns:
        tuple: (diff_tool, output_type, enable_lineno)
    """
    assert specify_type in [None, 'default', 'fullcode', 'diff', 'adaptive'], f"Unsupported output type: {specify_type}"

    parsed_format = _parse_edit_format(edit_format)
    base_format = parsed_format["base_format"]
    diff_tool = None
    enable_lineno = parsed_format["enable_lineno"]
    
    if base_format == "fullcode":
        # cannot specify diff-based output type for fullcode models
        return diff_tool, base_format, enable_lineno
    
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from data_gen.differ import UniDiffTool, MinUniDiffTool, MinContentDiffTool, BlockDiffTool
    if base_format == "minunidiff":
        diff_tool = MinUniDiffTool()
    elif base_format == "unidiff":
        diff_tool = UniDiffTool()
    else:
        diff_shape = parsed_format["diff_shape"]
        
        if base_format == "mincontentdiff":
            diff_tool = MinContentDiffTool(diff_shape=diff_shape)
        elif base_format == "contentdiff":
            diff_tool = MinContentDiffTool(diff_shape=diff_shape, min_context_lines=3)
        elif base_format == "blockdiff":
            diff_tool = BlockDiffTool(structure_type='block', diff_shape=diff_shape)
        elif base_format == "funcdiff":
            diff_tool = BlockDiffTool(structure_type='function', diff_shape=diff_shape)

    if not diff_tool:
        raise ValueError(f"Unsupported diff format: {edit_format}")
    
    if specify_type and specify_type != 'default':
        output_type = specify_type
    elif parsed_format["is_adaptive"]:
        output_type = 'adaptive'
    else:
        output_type = 'diff'
    
    return diff_tool, output_type, enable_lineno


def resolve_datasets(datasets: list):
    '''
    Return valid datasets with params
    '''
    # bechmarks for other languages need to be specified, such as 'humanevalfix-js'
    edit_datasets = ['humanevalfix-python', 'editeval', 'canitedit', 'aider']

    # datasets
    label_mappings = {
        'edit': edit_datasets,
    }
    valid_datasets = set(datasets)
    for label, ds_list in label_mappings.items():
        if label in valid_datasets:
            valid_datasets.remove(label)
            valid_datasets.update(ds_list)
    
    mappings = {}
    for ds in valid_datasets:
        mappings[ds] = {'temperature': 0.2, 'n_samples': 20}
    if 'aider' in valid_datasets:
        mappings['aider'] = {'temperature': 0.0, 'n_samples': 1}
    
    return mappings


def get_evaluator_class(dataset: str):
    if dataset == 'editeval':
        from benchmarks._editeval import EditEvalEvaluator as evaluator_class
    elif dataset == 'canitedit':
        from benchmarks._canitedit import CanItEditEvaluator as evaluator_class
    elif "humanevalfix" in dataset:
        from benchmarks._humanevalfix import HumanEvalFixEvaluator as evaluator_class
    elif dataset == "aider":
        from benchmarks._aider import AiderEvaluator as evaluator_class
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return evaluator_class
