"""
the most common diff format: unified diff
"""

import os
import tempfile
import subprocess

from .base_diff import BaseDiffTool


class UniDiffError(Exception):
    pass


class UniDiffTool(BaseDiffTool):
    def __init__(self, context_lines: int = 3, custom_patch: bool = True, strict_mode: bool = False):
        ''' 
        Parameters:
            context_lines (int): Number of lines of context to include in the diff.
            custom_patch (bool): Whether to use custom patch.
            strict_mode (bool): Whether to use strict mode.
        '''
        self.context_lines = context_lines
        self.custom_patch = custom_patch
        self.strict_mode = strict_mode


    def calculate_diff(self, text1: str, text2: str, **kwargs) -> str:
        '''
        Standard diff tool: diff -U
        '''
        if not text1 and not text2:
            return ""
        
        text1 = self._ensure_newline(text1)
        text2 = self._ensure_newline(text2)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as text1_file, \
            tempfile.NamedTemporaryFile(mode='w', delete=False) as text2_file:
            try:
                text1_file.write(text1)
                text2_file.write(text2)
                text1_file.flush()
                text2_file.flush()

                process = subprocess.run(
                    ['diff', f'-u{self.context_lines}', text1_file.name, text2_file.name],
                    capture_output=True,
                    text=True
                )

                # 0: same; 1: has diff; 2: error
                if process.returncode > 1:
                    raise RuntimeError(f"diff failed:\n{process.stdout}\n{process.stderr}")

                diff_output = process.stdout
                if diff_output:
                    # remove header
                    lines = diff_output.splitlines(keepends=True)
                    diff_output = ''.join(lines[2:])

            finally:
                os.unlink(text1_file.name)
                os.unlink(text2_file.name)

        return diff_output

    # -------------------------------------------------------- #

    def apply_diff(self, text: str, diff: str) -> str:
        text = self._ensure_newline(text)
        if not diff.strip():
            return text
        
        diff = self._ensure_newline(diff)
        # avoid pure-empty lines from LLM output
        diff_lines = diff.splitlines(keepends=True)
        for i, line in enumerate(diff_lines):
            if not line.strip() and not line.startswith(' '):
                diff_lines[i] = ' ' + line
        diff = ''.join(diff_lines)

        if self.custom_patch:
            return self.custom_patch_tool(text, diff)
        else:
            return self.standard_patch_tool(text, diff)
    

    def standard_patch_tool(self, text: str, diff: str) -> str:
        '''
        Standard patch tool: patch -u
        '''
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as text_file, \
             tempfile.NamedTemporaryFile(mode='w', delete=False) as diff_file:
            try:
                # Write original text and diff to files
                text_file.write(text)
                text_file.flush()

                diff_file.write(diff)
                diff_file.flush()

                # Run patch command with -u option; Set the fuzz factor to LINES for inexact matching
                result = subprocess.run(
                    ['patch', '-u', '-F', f'{self.context_lines}', text_file.name, '-i', diff_file.name],
                    capture_output=True,
                    text=True,
                    timeout=30  # Set a timeout to avoid hanging
                )

                if result.returncode != 0:
                    raise UniDiffError(f"patch failed:\n{result.stdout}\n{result.stderr}")
            
            except Exception:
                if self.strict_mode:
                    raise

            finally:
                # Read the patched file
                with open(text_file.name, 'r') as f:
                    patched_text = f.read()

                # remove all temporary files
                os.unlink(text_file.name)
                os.unlink(diff_file.name)

                orig_file = text_file.name + '.orig'
                if os.path.exists(orig_file):
                    os.unlink(orig_file)
                rej_file = text_file.name + '.rej'
                if os.path.exists(rej_file):
                    os.unlink(rej_file)

            return patched_text
    

    def custom_patch_tool(self, text: str, diff: str) -> str:
        '''
        Custom patch tool: tolerate some errors of LLM output
        '''
        diff_list = self.parse_diff(diff)
        if not diff_list:
            return text
        
        insert_dict = {}
        delete_positions = set()
        for numbers, content in diff_list:
            try:
                pos = numbers[0] - 1
                del_count = numbers[1]
                
                ins_pos = pos + 1
                insert_content = [x[1:] for x in content if x.startswith('+') or x.startswith(' ')]
                if insert_content:
                    if ins_pos in insert_dict:
                        raise UniDiffError("Conflict in the same block")
                    insert_dict[ins_pos] = insert_content
                
                if del_count > 0:
                    delete_positions.update(range(pos, pos + del_count))
            
            except Exception:
                if self.strict_mode:
                    raise

        lines = []
        if 0 in insert_dict:
            lines.extend(insert_dict[0])

        orig_lines = text.splitlines(keepends=True)
        for i, line in enumerate(orig_lines):
            if i not in delete_positions:
                lines.append(line)

            if i + 1 in insert_dict:
                lines.extend(insert_dict[i + 1])

        return ''.join(lines)


    def parse_diff(self, diff_text: str) -> list:
        '''
        For number-indexed diff formats
        '''
        diff_text = '\n' + diff_text.lstrip()
        if not diff_text.endswith('\n'):
            diff_text += '\n'

        diff_blocks = diff_text.split('\n@@')
        # Add the splited @@ back
        diff_blocks = ['@@' + block for block in diff_blocks[1:]]
        for i in range(len(diff_blocks) - 1):
            # Add the splited \n back except for the last block
            diff_blocks[i] = diff_blocks[i] + '\n'
        
        result = []
        for block in diff_blocks:
            if not block.strip():
                continue

            try:
                split_diff = block.splitlines(keepends=True)

                # header metadata
                first_line = split_diff[0]
                if first_line.count('@@') != 2:
                    raise UniDiffError

                numbers = []
                # @@ -a,b +c,d @@
                parts = first_line.replace('@@', '').strip().split(' ')
                if len(parts) != 2:
                    raise UniDiffError
                
                # -a,b
                nums = parts[0].split(',')
                if not nums[0].startswith('-'):
                    raise UniDiffError
                numbers.append(int(nums[0][1:]))

                if len(nums) == 1:
                    # 1 by default
                    numbers.append(1)
                elif len(nums) == 2:
                    numbers.append(int(nums[1]))
                else:
                    raise UniDiffError
                
                # +c,d
                nums = parts[1].split(',')
                if not nums[0].startswith('+'):
                    raise UniDiffError
                numbers.append(int(nums[0][1:]))
                
                if len(nums) == 1:
                    # 1 by default
                    numbers.append(1)
                elif len(nums) == 2:
                    numbers.append(int(nums[1]))
                else:
                    raise UniDiffError
                
                # check if numbers are valid
                if len(numbers) != 4 or any(num < 0 for num in numbers):
                    raise UniDiffError
                
                content = split_diff[1:]
                result.append((numbers, content))

            except Exception:
                if self.strict_mode:
                    raise UniDiffError(f"Invalid unified diff header: {first_line}")
                else:
                    # skip invalid diff block
                    continue

        return result


if __name__ == "__main__":
    from example import source_code, target_code

    diff_tool = UniDiffTool(strict_mode=True)
    diff = diff_tool.calculate_diff(source_code, target_code)
    print(diff)

    patched_code = diff_tool.apply_diff(source_code, diff)
    print(patched_code)

    assert diff_tool._ensure_newline(patched_code) == diff_tool._ensure_newline(target_code), "The patched code does not match the target code."
