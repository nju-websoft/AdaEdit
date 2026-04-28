'''
Unified diff -U0 with loose patch
'''

from .unidiff import UniDiffTool


class MinUniDiffTool(UniDiffTool):
    def __init__(self, custom_patch: bool = True, strict_mode: bool = False):
        # Initialize with no context lines
        super().__init__(0, custom_patch, strict_mode)


if __name__ == "__main__":
    from example import source_code, target_code

    diff_tool = MinUniDiffTool(strict_mode=True)
    diff = diff_tool.calculate_diff(source_code, target_code)
    print(diff)

    patched_code = diff_tool.apply_diff(source_code, diff)
    print(patched_code)

    assert diff_tool._ensure_newline(patched_code) == diff_tool._ensure_newline(target_code), "The patched code does not match the target code."
