import ast
from contextlib import nullcontext

import black
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


def syntax_check(code: str) -> bool:
    # only support Python 3
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError) as e:
        # print(repr(e))
        return False


class PythonParser:
    def __init__(self, lock=None) -> None:
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)

        self.lock = lock
    
    def remove_blank_lines(self, code):
        # remove all blank lines
        return '\n'.join([line for line in code.split('\n') if line.strip()])

    def traverse_comments_nodes(self, node, nodes_to_remove):
        if node.type == 'comment':
            nodes_to_remove.append(node)
        
        elif node.type == 'expression_statement' and len(node.children) == 1:
            # remove docstring
            child_type = node.children[0].type
            if child_type == 'string' or child_type == 'concatenated_string':
                nodes_to_remove.append(node)
        
        for child in node.children:
            self.traverse_comments_nodes(child, nodes_to_remove)

    def remove_code_by_nodes(self, code: bytes, nodes_to_remove: list):
        positions = []
        for node in nodes_to_remove:
            if isinstance(node, tuple):
                positions.append(node)
            else:
                positions.append((node.start_byte, node.end_byte))

        # order by start position
        positions.sort()

        # merge overlapping positions
        merged_positions = []
        for start, end in positions:
            if merged_positions and start <= merged_positions[-1][1]:
                # overlap, update the end position
                merged_positions[-1] = (merged_positions[-1][0], max(merged_positions[-1][1], end))
            else:
                # new interval
                merged_positions.append((start, end))

        # create a new byte array to store the new code
        new_code = bytearray()
        last_end = 0

        # joint the code
        for start, end in merged_positions:
            if start > last_end:
                new_code.extend(code[last_end:start])
            last_end = end

        # add the remaining code
        if last_end < len(code):
            new_code.extend(code[last_end:])

        return bytes(new_code).decode("utf-8", 'ignore')

    def remove_comments(self, code: str) -> str:
        """remove all comments in code, including docstring and inline comments"""
        try:
            code_byte = bytes(code, "utf-8")
            tree = self.parser.parse(code_byte)

            nodes_to_remove = []
            self.traverse_comments_nodes(tree.root_node, nodes_to_remove)
            return self.remove_code_by_nodes(code_byte, nodes_to_remove)
        except Exception as e:
            # print(f"Error removing comments: {repr(e)}")
            # if fail, return the original code
            return code

    def format_code(self, code: str) -> str:
        """Code formatter"""
        # use black to format the code: protect for concurrency
        with self.lock if self.lock is not None else nullcontext():
            try:
                # only support for Python 3 right now
                return black.format_str(code, mode=black.Mode(string_normalization=False))
            except Exception as e:
                # print(f"Error formatting code: {repr(e)}")
                # if fail, return the original code
                return code
        
    def normalize_code(self, code: str) -> str:
        """For comparison: remove all comments and format code"""
        code = self.remove_comments(code)
        code = self.format_code(code)

        return self.remove_blank_lines(code)
