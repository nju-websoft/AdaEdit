import importlib
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field


@dataclass
class LanguageConfig:
    """Language-specific configuration"""
    tree_sitter_module: str
    
    # structure level: block
    block_mode: Dict[str, Set[str]]
    # structure level: function
    function_mode: Dict[str, Set[str]]

    # decorator-like types
    decorator_like_types: Dict[str, str] = field(default_factory=dict)


class LanguageRegistry:
    _configs: Dict[str, LanguageConfig] = {}
    
    @classmethod
    def register(cls, language: str | List[str], config: LanguageConfig):
        """Register a new programming language."""
        if isinstance(language, str):
            language = [language]
        for item in language:
            cls._configs[item] = config
    
    @classmethod
    def get_config(cls, language: str) -> LanguageConfig:
        """Get language-specific configuration for a given language."""
        if language not in cls._configs:
            raise ValueError(f"Unsupported language: {language}. "
                           f"Supported languages: {', '.join(cls._configs.keys())}")
        return cls._configs[language]
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get all supported languages."""
        return list(cls._configs.keys())


# ============================================================================
# Python
# ============================================================================
LanguageRegistry.register('python', LanguageConfig(
    tree_sitter_module='tree_sitter_python',
    
    block_mode={
        # block types for BlockTree construction
        'block_types': {
            'if_statement', 'match_statement',
            'while_statement', 'for_statement', 
            'try_statement', 'with_statement',
            'function_definition', 
            'class_definition', 'decorated_definition',
        },
        # conditional block types for BlockTree construction
        'conditional_types': {},
        # coarse-grained nodes for diff generation
        'coarse_grained_types': {'class_definition', 'module'},
        # fine-grained nodes for diff generation
        'fine_grained_types': {
            'if_statement', 'match_statement',
            'while_statement', 'for_statement', 
            'try_statement', 'with_statement',
            'function_definition'
        }
    },
    
    function_mode={
        'block_types': {
            'function_definition', 'class_definition', 'decorated_definition',
        },
        'conditional_types': {},
        'coarse_grained_types': {'class_definition', 'module'},
        'fine_grained_types': {'function_definition'}
    },

    decorator_like_types={
        'decorated_definition': 'definition'
    }
))


# ============================================================================
# JavaScript
# ============================================================================
LanguageRegistry.register(['js', 'javascript'], LanguageConfig(
    tree_sitter_module='tree_sitter_javascript',
    
    block_mode={
        'block_types': {
            'if_statement', 'switch_statement', 'statement_block',
            'while_statement', 'for_statement', 'for_in_statement',
            'do_statement', 'try_statement', 'with_statement',
            'function_declaration', 'generator_function_declaration', 'method_definition',
            'function_expression', 'arrow_function', 'generator_function',
            'class_declaration', 'class_static_block'
        },
        'conditional_types': {
            'statement_block': {'program', 'statement_block'}
        },
        'coarse_grained_types': {
            'class_declaration', 'program'
        },
        'fine_grained_types': {
            'if_statement', 'switch_statement',
            'while_statement', 'for_statement', 'for_in_statement',
            'do_statement', 'try_statement', 'with_statement',
            'function_declaration', 'generator_function_declaration', 'arrow_function',
            'method_definition', 'function_expression', 'generator_function',
            'class_static_block', 'statement_block'
        }
    },
    
    function_mode={
        'block_types': {
            'function_declaration', 'generator_function_declaration', 'method_definition',
            'function_expression', 'arrow_function', 'generator_function',
            'class_declaration'
        },
        'coarse_grained_types': {
            'class_declaration', 'program'
        },
        'fine_grained_types': {
            'function_declaration', 'generator_function_declaration', 'method_definition',
            'function_expression', 'arrow_function', 'generator_function'
        }
    }
))



# ============================================================================
# Java: Not Validated
# ============================================================================
LanguageRegistry.register('java', LanguageConfig(
    tree_sitter_module='tree_sitter_java',
    
    block_mode={
        'block_types': {
            'if_statement', 'switch_expression', 'synchronized_statement', 
            'while_statement', 'for_statement', 'enhanced_for_statement',
            'do_statement', 'try_statement', 'try_with_resources_statement', 'block',
            'method_declaration', 'constructor_declaration',
            'class_declaration', 'interface_declaration', 'enum_declaration',
            'record_declaration', 'annotation_type_declaration', 'static_initializer'
        },
        'conditional_types': {
            # allowed parents
            'block': {'class_body', 'enum_body'}
        },
        'coarse_grained_types': {
            'class_declaration', 'interface_declaration', 'enum_declaration',
            'record_declaration', 'annotation_type_declaration',
            'program'
        },
        'fine_grained_types': {
            'if_statement', 'switch_expression', 'do_statement', 
            'while_statement', 'for_statement', 'enhanced_for_statement',
            'try_statement', 'try_with_resources_statement', 'synchronized_statement', 
            'static_initializer', 'block', 'method_declaration', 'constructor_declaration'
        }
    },
    
    function_mode={
        'block_types': {
            'method_declaration', 'constructor_declaration',
            'class_declaration', 'interface_declaration', 'enum_declaration',
            'record_declaration', 'annotation_type_declaration'
        },
        'coarse_grained_types': {
            'class_declaration', 'interface_declaration', 'enum_declaration',
            'record_declaration', 'annotation_type_declaration',
            'program'
        },
        'fine_grained_types': {
            'method_declaration', 'constructor_declaration'
        }
    }
))


# ============================================================================
# C++: Not Validated
# ============================================================================
LanguageRegistry.register('cpp', LanguageConfig(
    tree_sitter_module='tree_sitter_cpp',
    
    block_mode={
        'block_types': {
            'if_statement', 'switch_statement', 'while_statement', 'compound_statement',
            'for_statement', 'do_statement', 'for_range_loop', 'try_statement', 
            'function_definition', 'enum_specifier',
            'class_specifier', 'struct_specifier', 'union_specifier', 'namespace_definition',  
            'template_declaration', 'friend_declaration', 'export_declaration'
        },
        'conditional_types': {
            'compound_statement': {'compound_statement', 'field_declaration_list'}
        },
        'coarse_grained_types': {
            'class_specifier', 'struct_specifier', 'union_specifier', 'namespace_definition',
            'translation_unit'
        },
        'fine_grained_types': {
            'if_statement', 'switch_statement', 'while_statement', 
            'for_statement', 'do_statement', 'for_range_loop', 'try_statement', 
            'function_definition', 'enum_specifier', 'compound_statement'
        }
    },
    
    function_mode={
        'block_types': {
            'function_definition', 'enum_specifier',
            'class_specifier', 'struct_specifier', 'union_specifier', 'namespace_definition',
            'template_declaration', 'friend_declaration', 'export_declaration'
        },
        'coarse_grained_types': {
            'class_specifier', 'struct_specifier', 'union_specifier', 'namespace_definition',
            'translation_unit'
        },
        'fine_grained_types': {'function_definition', 'enum_specifier'}
    },

    decorator_like_types={
        'template_declaration': None,  # None denotes the first BlockNode
        'friend_declaration': None,
        'export_declaration': None
    }
))


# ============================================================================
# Rust: Not Validated
# ============================================================================
LanguageRegistry.register('rust', LanguageConfig(
    tree_sitter_module='tree_sitter_rust',
    
    block_mode={
        'block_types': {
            'if_expression', 'match_expression',
            'while_expression', 'for_expression', 'loop_expression',
            'function_item',
            'impl_item', 'struct_item', 'enum_item', 'trait_item', 
            'union_item', 'mod_item', 'foreign_mod_item'
        },
        'coarse_grained_types': {
            'impl_item', 'trait_item',
            'mod_item', 'foreign_mod_item',
            'source_file'
        },
        'fine_grained_types': {
            'if_expression', 'match_expression',
            'while_expression', 'for_expression', 'loop_expression',
            'function_item',
            'struct_item', 'enum_item', 'union_item'
        }
    },
    
    function_mode={
        'block_types': {
            'function_item',
            'impl_item', 'struct_item', 'enum_item', 'trait_item', 
            'union_item', 'mod_item', 'foreign_mod_item'
        },
        'coarse_grained_types': {
            'impl_item', 'trait_item', 
            'mod_item', 'foreign_mod_item',
            'source_file'
        },
        'fine_grained_types': {
            'function_item', 
            'struct_item', 'enum_item', 'union_item'
        }
    },
))


# ============================================================================
# Go: Not Validated
# ============================================================================
LanguageRegistry.register('go', LanguageConfig(
    tree_sitter_module='tree_sitter_go',
    
    block_mode={
        'block_types': {
            'if_statement', 'expression_switch_statement', 'type_switch_statement',
            'for_statement', 'select_statement', 'block',
            'function_declaration', 'method_declaration', 'func_literal',
        },
        'conditional_types': {
            'block': {'block'}
        },
        'coarse_grained_types': {
            'source_file'
        },
        'fine_grained_types': {
            'if_statement', 'expression_switch_statement', 'type_switch_statement',
            'for_statement', 'select_statement', 'block',
            'function_declaration', 'method_declaration', 'func_literal'
        }
    },
    
    function_mode={
        'block_types': {
            'function_declaration', 'method_declaration', 'func_literal',
        },
        'coarse_grained_types': {
            'source_file'
        },
        'fine_grained_types': {
            'function_declaration', 'method_declaration', 'func_literal'
        }
    }
))


# ============================================================================
# BlockNode and BlockTree
# ============================================================================

class BlockNode:
    def __init__(self, parent, start_lineno: int, end_lineno: int, ast_type: str):
        self.parent = parent
        self.start_lineno = start_lineno
        self.end_lineno = end_lineno
        
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

        self.children = []
        self.ast_type = ast_type
    
    def add_child(self, child_node, pos=None):
        assert child_node.start_lineno >= self.start_lineno and \
               child_node.end_lineno <= self.end_lineno, \
               "Unexpected parsing results"

        if pos is None:
            self.children.append(child_node)
        else:
            self.children.insert(pos, child_node)
    
    def __repr__(self):
        return f"BlockNode(type={self.ast_type}, start_lineno={self.start_lineno}, " \
               f"end_lineno={self.end_lineno})"


class BlockTree:
    def __init__(self, code: str, structure_type: str = 'block', language: str = 'python'):
        """
        Args:
            code: source code
            structure_type: 'block' or 'function'
            language: support python, java, cpp, javascript, rust, go
        """
        from tree_sitter import Language, Parser
        
        self.language = language
        self.structure_type = structure_type

        self.GENERAL_BLOCK_TYPE = 'general_code'
        
        # get language config
        self.lang_config = LanguageRegistry.get_config(language)
        
        # import tree-sitter-xxx
        try:
            tslang = importlib.import_module(self.lang_config.tree_sitter_module)
        except ImportError:
            raise ImportError(f"Please install {self.lang_config.tree_sitter_module}: "
                            f"pip install {self.lang_config.tree_sitter_module}")
        
        self.parser = Parser(Language(tslang.language()))
        self.decorator_like_types = self.lang_config.decorator_like_types
        
        # load specific configs
        self._load_mode_config()
        
        # parse code
        self.code_lines = code.splitlines(keepends=True)
        self.root_node = self.build_tree(code.rstrip())
    
    def _load_mode_config(self):
        """load configs based on structure_type"""
        if self.structure_type == 'block':
            mode_config = self.lang_config.block_mode
        elif self.structure_type == 'function':
            mode_config = self.lang_config.function_mode
        else:
            raise ValueError(f"Unsupported structure_type: {self.structure_type}. "
                           f"Use 'block' or 'function'.")
        
        self.block_types = mode_config['block_types'].copy()
        self.conditional_types = mode_config.get('conditional_types', {}).copy()
        self.coarse_grained_types = mode_config['coarse_grained_types'].copy()
        self.fine_grained_types = mode_config['fine_grained_types'].copy()
        
        # add special types
        self.fine_grained_types.add(self.GENERAL_BLOCK_TYPE)

    # ============= Below are BlockTree construction =================
    def create_node(self, parent, ast_node, node_type_override=None, index=None):
        start_lineno = ast_node.start_point[0]
        end_lineno = ast_node.end_point[0]
        node_type = node_type_override if node_type_override else ast_node.type
        node = BlockNode(parent, start_lineno, end_lineno, node_type)
        if parent:
            parent.add_child(node, index)
        return node
    
    def build_tree(self, code: str) -> Optional[BlockNode]:
        try:
            code_byte = bytes(code, "utf-8")
            tree = self.parser.parse(code_byte)
            ast_root = tree.root_node
            
            # The root_node of our BlockTree is always a virtual node covering the whole file
            self.root_type = ast_root.type
            root_node = BlockNode(None, 0, len(self.code_lines) - 1, self.root_type)
            
            for child_ast in ast_root.children:
                self._build_recursive(child_ast, root_node)

            self._fill_gaps(root_node)
            return root_node

        except Exception as e:
            print(f"Error in building block tree: {repr(e)}")
            print(repr(code))
            raise
            return None
    
    def _build_recursive(self, current_ast_node, parent_block_node: BlockNode):
        node_type = current_ast_node.type
        
        # --- 1. Handle decorator-like nodes (e.g., Python decorators, C++ templates) ---
        if node_type in self.decorator_like_types:
            field_name = self.decorator_like_types[node_type]
            definition_child = None
            if field_name:
                # Case 1: The real definition is in a named field (e.g., Python)
                definition_child = current_ast_node.child_by_field_name(field_name)
            else:
                # Case 2: The real definition is the first child that is a block (e.g., C++ template)
                for child in current_ast_node.children:
                    if child.type in self.block_types:
                        definition_child = child
                        break
            
            if definition_child:
                # Create a block for the definition, but use the decorator's full span
                new_block_node = self.create_node(
                    parent_block_node,
                    current_ast_node, # Use wrapper node for line numbers
                    node_type_override=definition_child.type # But use inner node's type
                )
                # Recurse into the *actual* definition's children
                for grand_child_ast in definition_child.children:
                    self._build_recursive(grand_child_ast, new_block_node)
                self._fill_gaps(new_block_node)
                return # Stop processing this branch further

        # --- 2. Handle regular nodes ---
        newly_created_block = None
        new_parent_for_children = parent_block_node
        is_valid_block = node_type in self.block_types

        # --- 2a. Check for conditional block types (e.g., Java's 'block') ---
        if is_valid_block and node_type in self.conditional_types:
            allowed_parents = self.conditional_types[node_type]
            parent_ast_type = current_ast_node.parent.type if current_ast_node.parent else ""
            if parent_ast_type not in allowed_parents:
                is_valid_block = False # Invalidate this block if parent doesn't match

        # --- 2b. If it's a valid block, create a BlockNode ---
        if is_valid_block:
            new_block_node = self.create_node(parent_block_node, current_ast_node)
            newly_created_block = new_block_node
            new_parent_for_children = new_block_node

        # --- 3. Recurse into children ---
        for child_ast_node in current_ast_node.children:
            self._build_recursive(child_ast_node, new_parent_for_children)

        # --- 4. Fill gaps if a new block was created at this level ---
        if newly_created_block:
            self._fill_gaps(newly_created_block)

    def _fill_gaps(self, parent_node: BlockNode):
        """Fill gaps between child blocks with 'general_code' blocks."""
        if parent_node.ast_type not in self.coarse_grained_types:
            return
        
        if not parent_node.children:
            return

        sorted_children = sorted(parent_node.children, key=lambda n: n.start_lineno)
        
        new_children_list = []
        cursor_line = parent_node.start_lineno
        parent_end_line = parent_node.end_lineno

        for child_node in sorted_children:
            child_start_line = child_node.start_lineno
            
            if cursor_line < child_start_line:
                gap_block = self._create_general_block(parent_node, cursor_line, child_start_line - 1)
                if gap_block:
                    new_children_list.append(gap_block)
            
            new_children_list.append(child_node)
            cursor_line = child_node.end_lineno + 1
            
        if cursor_line <= parent_end_line:
            gap_block = self._create_general_block(parent_node, cursor_line, parent_end_line)
            if gap_block:
                new_children_list.append(gap_block)
        
        parent_node.children = new_children_list

    def _create_general_block(self, parent: BlockNode, start_line: int, end_line: int) -> Optional[BlockNode]:
        """Create a 'general_code' block, trimming empty lines from start and end."""
        true_start, true_end = start_line, end_line

        while true_start <= true_end and not self.code_lines[true_start].strip():
            true_start += 1
        
        while true_end >= true_start and not self.code_lines[true_end].strip():
            true_end -= 1

        if true_start > true_end:
            return None

        node = BlockNode(parent, true_start, true_end, self.GENERAL_BLOCK_TYPE)
        return node
    
    # ============================================================================
    # The following methods are for querying the tree. They are language-agnostic
    # and operate on the constructed BlockNode tree, so they don't need changes.
    # ============================================================================

    def print_block_tree(self, node_list: Optional[List[BlockNode]] = None):
        def traverse(blocks, prefix):
            for i, block in enumerate(blocks):
                connector = "└── " if i == len(blocks) - 1 else "├── "
                print(f"{prefix}{connector}{block.ast_type} "
                      f"({block.start_lineno}-{block.end_lineno})")
                if block.children:
                    extension = "│   " if i < len(blocks) - 1 else "    "
                    traverse(block.children, prefix + extension)
        
        if node_list is None:
            node_list = [self.root_node] if self.root_node else []
        traverse(node_list, "")

    def find_smallest_block_for_line(self, line_no: int, search_node: BlockNode) -> BlockNode:
        for child in search_node.children:
            if child.start_lineno <= line_no <= child.end_lineno:
                return self.find_smallest_block_for_line(line_no, child)
        return search_node

    def find_optimal_blocks(self, start_lineno: int, end_lineno: int) -> tuple[tuple[int, int], list[BlockNode]]:
        if start_lineno > end_lineno:
            raise ValueError("start_lineno cannot be greater than end_lineno")
        
        result_blocks = []
        cursor = start_lineno

        while cursor <= end_lineno:
            if not self.code_lines[cursor].strip():
                cursor += 1
                continue

            smallest_block = self.find_smallest_block_for_line(cursor, self.root_node)
            result_blocks.append(smallest_block)
            cursor = smallest_block.end_lineno + 1
        
        content_range = self.calculate_content_range(result_blocks) if result_blocks else (start_lineno, end_lineno)
        return content_range, result_blocks
    
    def calculate_content_range(self, block_list: List[BlockNode]) -> tuple[int, int]:
        # if not block_list:
        #     return -1, -1
        return min(b.start_lineno for b in block_list), max(b.end_lineno for b in block_list)

    def _recursive_find_container(self, current_node: BlockNode, start_lineno: int, end_lineno: int) -> Optional[BlockNode]:
        if not (current_node.start_lineno <= start_lineno and current_node.end_lineno >= end_lineno):
            return None

        for child in current_node.children:
            found_in_child = self._recursive_find_container(child, start_lineno, end_lineno)
            if found_in_child:
                return found_in_child
        return current_node
    
    def find_smallest_containing_block(self, start_lineno: int, end_lineno: int) -> Optional[BlockNode]:
        if start_lineno > end_lineno:
            raise ValueError("start_lineno cannot be greater than end_lineno")
        if not self.root_node:
            return None
        return self._recursive_find_container(self.root_node, start_lineno, end_lineno)
    
    def clean_blocks(self, block_list: list[BlockNode]):
        if len(block_list) <= 1:
            return block_list
        
        initial_range = self.calculate_content_range(block_list)

        # Pruning: remove blocks fully contained within others in the list
        pruned_blocks: List[BlockNode] = []
        sorted_by_size = sorted(block_list, key=lambda b: b.end_lineno - b.start_lineno, reverse=True)
        for current_block in sorted_by_size:
            if not any(b.start_lineno <= current_block.start_lineno and b.end_lineno >= current_block.end_lineno for b in pruned_blocks):
                pruned_blocks.append(current_block)

        # Merging: if all children of a parent are in the list, replace them with the parent
        current_blocks = set(pruned_blocks)
        while True:
            nodes_to_add = set()
            nodes_to_remove = set()
            
            potential_parents = {node.parent for node in current_blocks if node.parent}
            if not potential_parents:
                break

            sorted_parents = sorted(list(potential_parents), key=lambda p: p.depth, reverse=True)
            for parent in sorted_parents:
                children_of_parent_in_set = {child for child in parent.children if child in current_blocks}
                
                if not children_of_parent_in_set:
                    continue

                # Check if all children of the parent that exist are in our current set
                if set(parent.children) == children_of_parent_in_set and \
                   parent.start_lineno >= initial_range[0] and parent.end_lineno <= initial_range[1]:
                    nodes_to_add.add(parent)
                    nodes_to_remove.update(children_of_parent_in_set)
            
            if not nodes_to_add:
                break
            
            current_blocks.difference_update(nodes_to_remove)
            current_blocks.update(nodes_to_add)

        optimal_blocks = sorted(list(current_blocks), key=lambda b: b.start_lineno)
        
        current_range = self.calculate_content_range(optimal_blocks)
        assert current_range == initial_range, 'Error in cleaning blocks: content range changed'

        return optimal_blocks


if __name__ == '__main__':
    from example import source_code, target_code
    tree = BlockTree(source_code, structure_type='function', language='python')
    tree.print_block_tree()
