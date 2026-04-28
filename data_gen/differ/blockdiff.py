import time
import collections
from typing import List, Dict, Set

from .contentdiff import ContentDiffTool
from .minunidiff import MinUniDiffTool
from .block_tree import BlockNode, BlockTree


class BlockDiffTool(ContentDiffTool):
    def __init__(self, structure_type='block', diff_shape='standard', strict_mode=False):
        """
        structure_type: block, function
        """
        assert structure_type in ['block', 'function'], 'Invalid structure type'

        super().__init__(0, diff_shape, strict_mode)
        self.structure_type = structure_type
        self.min_unidiff_tool = MinUniDiffTool(strict_mode=True)


    def calculate_diff(self, text1: str, text2: str, **kwargs) -> str:
        # Stage 1: unified diff -U0
        # stime = time.time()
        unidiff = self.min_unidiff_tool.calculate_diff(text1, text2)
        # print('Calculate text diff:', time.time() - stime)
        if not unidiff.strip():
            return ''
        
        original_text = self._ensure_newline(text1)
        self.source_lines = original_text.splitlines(keepends=True)
        diff_list = self.min_unidiff_tool.parse_diff(unidiff)

        if not original_text:
            # special case: empty original text
            content_diff_blocks = [((0, -1), diff_list)]
        else:
            # original states
            self.compare_source_lines = [''.join(x.split()) for x in self.source_lines if x.strip()]
            
            # Stage 2: create AST-based code block tree
            # stime = time.time()
            self.block_tree = BlockTree(original_text, self.structure_type, kwargs['lang'])
            # print('Calculate block tree:', time.time() - stime)
            # self.block_tree.print_block_tree()    # debug

            # Stage 3: initialize blocks for anchor
            content_diff_blocks = self.convert_to_diff_blocks(diff_list)

            # Stage 4: expand and merge anchor blocks
            content_diff_blocks = self.transform_blocks(content_diff_blocks)

            # prepare for restore: remove AST information
            content_diff_blocks = [(anchor_range, diff_info) for anchor_range, diff_info, _ in content_diff_blocks]

        # Stage 5: convert to the defined diff format
        ret = self.restore_diff_text(content_diff_blocks)

        return ret


    def convert_to_diff_blocks(self, diff_list: list) -> list:
        '''
        Convert unified diff to content-addressed diff hunks in block level
        '''
        content_diff_blocks = []    # [(anchor_range, [((del_pos, del_count, ins_pos, ins_count), content)], min_blocks)]

        max_line_index = len(self.source_lines) - 1
        offset = 0
        for (numbers, content) in diff_list:
            # each diff block
            del_pos, del_count, ins_pos, ins_count = numbers
            # adjust the position based on the original text
            ins_pos -= offset
            offset += ins_count - del_count

            if del_count > 0:
                # Convert to zero-based index
                start_pos = del_pos - 1
                end_pos = start_pos + del_count - 1
            else:
                # the insertion position
                start_pos = min(max_line_index, ins_pos - 1)
                end_pos = start_pos
            
            # find the minimal blocks
            anchor_range, min_blocks = self.block_tree.find_optimal_blocks(start_pos, end_pos)
            content_diff_blocks.append((anchor_range, [((del_pos, del_count, ins_pos, ins_count), content)], min_blocks))

        return content_diff_blocks
    

    def merge_blocks_by_position(self, content_diff_blocks: list):
        """
        Merge adjacent blocks by position
        """
        if len(content_diff_blocks) <= 1:
            return content_diff_blocks

        new_blocks = []

        index = 0
        content_diff_blocks.sort(key=lambda x: x[0])
        while index < len(content_diff_blocks):
            current_block = content_diff_blocks[index]
            current_left_pos, current_right_pos = current_block[0]

            start_index = index
            index += 1
            while index < len(content_diff_blocks):
                next_block = content_diff_blocks[index]
                # place in one state
                next_left_pos, next_right_pos = next_block[0]

                # Check if the next block can be merged with the current block
                if self.is_overlapping(current_left_pos, current_right_pos, next_left_pos, next_right_pos, self.max_merge_gap):
                    # Merge the two blocks
                    current_left_pos = min(current_left_pos, next_left_pos)
                    current_right_pos = max(current_right_pos, next_right_pos)
                    index += 1
                else:
                    # end this block
                    break

            # Add the merged block to the new list
            if index == start_index + 1:
                # only one
                new_blocks.append(current_block)
            else:
                pos_blocks = []
                for x in content_diff_blocks[start_index:index]:
                    pos_blocks.extend(x[1])
                # indicate merged blocks: set block_list to None
                new_blocks.append(((current_left_pos, current_right_pos), pos_blocks, None))

        # re-calculate for merged blocks
        for i, (anchor_range, diff_info, block_list) in enumerate(new_blocks):
            if block_list is None:
                anchor_range, min_blocks = self.block_tree.find_optimal_blocks(*anchor_range)
                new_blocks[i] = (anchor_range, diff_info, min_blocks)
        
        return new_blocks
    

    def count_non_empty_lines(self, start_pos: int, end_pos: int) -> int:
        return sum(1 for i in range(start_pos, end_pos + 1) if self.source_lines[i].strip())
    

    def expand_blocks(self, content_diff_blocks: list):
        '''
        expand blocks to guarantee unambiguous patching
        '''
        for i, (anchor_range, diff_info, _) in enumerate(content_diff_blocks):
            start_pos, end_pos = anchor_range

            # ensure the block can be located only once
            if self.is_content_unique(start_pos, end_pos):
                continue
            
            # expand anchor for unique content
            while True:
                # add a node in the left/right side
                scope_block = self.block_tree.find_smallest_containing_block(start_pos, end_pos)

                # the minimal underexplored scope
                while scope_block.start_lineno == start_pos and scope_block.end_lineno == end_pos:
                    scope_block = scope_block.parent
                
                # find the left neighbor
                left_neighbor = None
                cursor = start_pos - 1
                if cursor >= scope_block.start_lineno:
                    while cursor >= scope_block.start_lineno:
                        if self.block_tree.code_lines[cursor].strip():
                            left_neighbor = self.block_tree.find_smallest_block_for_line(cursor, scope_block)
                            break
                        cursor -= 1
                
                # find the right neighbor
                right_neighbor = None
                cursor = end_pos + 1
                if cursor <= scope_block.end_lineno:
                    while cursor <= scope_block.end_lineno:
                        if self.block_tree.code_lines[cursor].strip():
                            right_neighbor = self.block_tree.find_smallest_block_for_line(cursor, scope_block)
                            break
                        cursor += 1
                
                if left_neighbor is None and right_neighbor is None:
                    break

                expand_left = False
                if right_neighbor is None:
                    expand_left = True
                elif left_neighbor is not None:
                    # core logic: whether the left neighbor contains the right neighbor
                    left_contains_right = (left_neighbor.start_lineno <= right_neighbor.start_lineno and left_neighbor.end_lineno >= right_neighbor.end_lineno)
                    if not left_contains_right:
                        expand_left = True
                
                if expand_left:
                    start_pos = min(left_neighbor.start_lineno, start_pos)
                    end_pos = max(left_neighbor.end_lineno, end_pos)
                else: 
                    start_pos = min(right_neighbor.start_lineno, start_pos)
                    end_pos = max(right_neighbor.end_lineno, end_pos)

                if self.is_content_unique(start_pos, end_pos):
                    break
            
            # update
            anchor_range, min_blocks = self.block_tree.find_optimal_blocks(start_pos, end_pos)
            content_diff_blocks[i] = (anchor_range, diff_info, min_blocks)


    def merge_blocks_by_ast(self, content_diff_blocks: list):
        '''
        multiple distinct hunks fall under a single fine-grained node
        '''
        # preprocess
        initial_unique_nodes: Set[BlockNode] = set()
        node_to_list_indices: Dict[BlockNode, Set[int]] = collections.defaultdict(set)
        for i, (_, _, block_list) in enumerate(content_diff_blocks):
            for node in block_list:
                initial_unique_nodes.add(node)
                node_to_list_indices[node].add(i)
        
        if not initial_unique_nodes:
            return

        ancestor_to_unique_descendants: Dict[BlockNode, Set[BlockNode]] = collections.defaultdict(set)
        for node in initial_unique_nodes:
            current = node
            while current:
                ancestor_to_unique_descendants[current].add(node)
                current = current.parent

        # iterative merge
        current_unique_nodes = initial_unique_nodes.copy()
        while True:
            merge_candidates: Dict[BlockNode, Set[BlockNode]] = collections.defaultdict(set)
            sorted_nodes = sorted(current_unique_nodes, key=lambda n: n.depth, reverse=True)
            
            for node in sorted_nodes:
                parent = node.parent
                while parent:
                    merge_candidates[parent].add(node)
                    parent = parent.parent
            
            nodes_to_add: Set[BlockNode] = set()
            nodes_to_remove: Set[BlockNode] = set()
            sorted_ancestors = sorted(merge_candidates.keys(), key=lambda n: n.depth, reverse=True)
            
            processed_in_this_pass = set()
            for ancestor in sorted_ancestors:
                current_descendants = {desc for desc in merge_candidates[ancestor] if desc not in processed_in_this_pass}

                if len(current_descendants) >= 2:
                    # fine-grained ancestor node: merge if there are multiple descendants
                    is_merge = ancestor.ast_type in self.block_tree.fine_grained_types
                    # coarse-grained ancestor node: only merge if the anchor lines cover all non-empty lines
                    if not is_merge and ancestor.ast_type in self.block_tree.coarse_grained_types:
                        anchor_line_nums = sum(self.count_non_empty_lines(desc.start_lineno, desc.end_lineno) for desc in current_descendants)
                        line_threshold = self.count_non_empty_lines(ancestor.start_lineno, ancestor.end_lineno)
                        if anchor_line_nums >= line_threshold:
                            is_merge = True
                    
                    if is_merge:
                        # merge
                        nodes_to_add.add(ancestor)
                        nodes_to_remove.update(current_descendants)
                        processed_in_this_pass.update(current_descendants)

            if not nodes_to_add:
                break
                
            current_unique_nodes.difference_update(nodes_to_remove)
            current_unique_nodes.update(nodes_to_add)

        # append new nodes
        added_nodes = current_unique_nodes - initial_unique_nodes
        if not added_nodes:
            return

        additions_map: Dict[int, List[BlockNode]] = collections.defaultdict(list)
        
        for parent_node in added_nodes:
            original_descendants = ancestor_to_unique_descendants.get(parent_node, set())
            
            relevant_indices = set()
            for desc in original_descendants:
                relevant_indices.update(node_to_list_indices.get(desc, set()))
                
            for list_index in relevant_indices:
                additions_map[list_index].append(parent_node)
        
        # update content_diff_blocks
        for list_index, nodes_to_append in additions_map.items():
            _, diff_info, block_list = content_diff_blocks[list_index]
            block_list.extend(nodes_to_append)
            anchor_range = self.block_tree.calculate_content_range(block_list)
            content_diff_blocks[list_index] = (anchor_range, diff_info, block_list)


    def transform_blocks(self, content_diff_blocks: list):
        '''
        (Merge) -> Expand -> AST Merge -> (Merge)
        '''
        content_diff_blocks = self.merge_blocks_by_position(content_diff_blocks)

        self.expand_blocks(content_diff_blocks)

        self.merge_blocks_by_ast(content_diff_blocks)

        content_diff_blocks = self.merge_blocks_by_position(content_diff_blocks)

        return content_diff_blocks


if __name__ == "__main__":
    from example import source_code, target_code

    diff_tool = BlockDiffTool("function", "standard", strict_mode=True)
    diff = diff_tool.calculate_diff(source_code, target_code, lang='python')
    print(diff)

    patched_code = diff_tool.apply_diff(source_code, diff)
    print(patched_code)

    assert diff_tool._ensure_newline(patched_code) == diff_tool._ensure_newline(target_code), "The patched code does not match the target code."

    # import os
    # with open('test0.py', 'r') as f:
    #     source_code = f.read()

    # with open('test1.py', 'r') as f:
    #     target_code = f.read()
    
    # diff_tool = BlockDiffTool("block", "standard", strict_mode=True)
    # stime = time.time()
    # diff = diff_tool.calculate_diff(source_code, target_code, lang='python')
    # print('Total:', time.time() - stime)
    # print(diff)
