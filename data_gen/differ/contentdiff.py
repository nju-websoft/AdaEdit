import re
from typing import Tuple, Dict, Set

from .base_diff import BaseDiffTool


class ContentDiffError(Exception):
    pass

class NoMatchError(ContentDiffError):
    pass

class MultipleMatchError(ContentDiffError):
    pass


class ContentDiffTool(BaseDiffTool):
    '''
    Content-addressed diff tool: with the same patch
    '''
    def __init__(self, max_merge_gap, diff_shape='standard', strict_mode=False):
        '''
        Parameters:
            max_merge_gap (int): Maximum gap between lines to be considered for merging (1 means close lines, 0 means no merging unless overlap).
            diff_shape (str): Shape of the diff, can be 'standard', 'interlaced', or 'search_replace'.
            strict_mode (bool): Whether to use strict mode for patch.
        '''
        assert max_merge_gap >= 0, "Maximum merge gap must be non-negative."
        assert diff_shape in ['standard', 'interlaced', 'search_replace'], 'Invalid diff shape'

        self.max_merge_gap = max_merge_gap
        self.strict_mode = strict_mode

        # Define the split header for content-addressed diff formats
        self.SPLIT_HEADER = '@@ ... @@\n'
        self.diff_shape = diff_shape
        if self.diff_shape == 'search_replace':
            self.search_splitter = "<<<<<<< SEARCH\n"
            self.replace_splitter = ">>>>>>> REPLACE\n"
            self.seperator = "=======\n"
            self.pattern = re.compile(r'^<<<<<<< SEARCH$(.*?)^=======$(.*?)^>>>>>>> REPLACE$', re.MULTILINE | re.DOTALL)
        
        self.source_lines = None
        self.compare_source_lines = None

    
    def is_content_unique(self, start_pos, end_pos) -> bool:
        '''
        Judge if the content from start_pos to end_pos is unique in the source_lines
        '''
        if start_pos == 0 and end_pos == len(self.source_lines) - 1:
            return True

        # do not rely on blank lines and indentation
        target_content = [''.join(line.split()) for line in self.source_lines[start_pos : end_pos + 1] if line.strip()]
        if not target_content:
            return False
        
        content_length = len(target_content)

        # sliding window to count occurrences
        found_count = 0
        for i in range(len(self.compare_source_lines) - content_length + 1):
            for j in range(content_length):
                if self.compare_source_lines[i + j] != target_content[j]:
                    break
            else:
                found_count += 1
                if found_count > 1:
                    return False

        return True
    

    def is_overlapping(self, start1, end1, start2, end2, offset):
        '''
        Check if two ranges are overlapping
        '''
        return (end1 >= start2 - offset and start1 <= end2 + offset)
    

    def restore_diff_text(self, content_diff_blocks):
        '''
        [(anchor_range, [((del_pos, del_count, ins_pos, ins_count), content)])]
        '''
        diff_text = ''
        sorted_hunks = sorted(content_diff_blocks, key=lambda x: x[0])

        for anchor_range, pos_blocks in sorted_hunks:
            if self.diff_shape == 'interlaced':
                hunk = self._generate_interleaved_contentdiff(anchor_range, pos_blocks)

            elif self.diff_shape == 'standard':
                hunk = self._generate_standard_contentdiff(anchor_range, pos_blocks)

            elif self.diff_shape == 'search_replace':
                hunk = self._generate_search_replace_contentdiff(anchor_range, pos_blocks)
            
            diff_text += hunk

        return diff_text


    def _generate_standard_contentdiff(self, anchor_range, pos_blocks):
        '''
        diff format with only + and -
        '''
        insert_dict, del_positions, loop_start, loop_end = self._preprocess_pos_blocks(anchor_range, pos_blocks)
        
        original_block = ['-' + self.source_lines[i] for i in range(loop_start, loop_end + 1)]
        
        new_block = []
        for i in range(loop_start, loop_end + 1):
            if i in insert_dict:
                new_block.extend(insert_dict[i])
            if i not in del_positions:
                new_block.append('+' + self.source_lines[i])
        if loop_end + 1 in insert_dict:
            new_block.extend(insert_dict[loop_end + 1])
        
        original_block, new_block = self._trim_common_blank_lines(original_block, new_block)

        # check
        original_lines = [line[1:] for line in original_block]
        new_lines = [line[1:] for line in new_block]
        if original_lines == new_lines:
            return ''

        return self.SPLIT_HEADER + ''.join(original_block) + ''.join(new_block)


    def _generate_interleaved_contentdiff(self, anchor_range, pos_blocks):
        '''
        Generate interleaved content diff: -, +, and unchanged lines
        '''
        insert_dict, del_positions, loop_start, loop_end = self._preprocess_pos_blocks(anchor_range, pos_blocks)
    
        diff_lines = []
        for i in range(loop_start, loop_end + 1):
            # deletion first
            if i in del_positions:
                diff_lines.append('-' + self.source_lines[i])
            
            if i in insert_dict:
                diff_lines.extend(insert_dict[i])

            if i not in del_positions:
                diff_lines.append(' ' + self.source_lines[i])

        if loop_end + 1 in insert_dict:
            diff_lines.extend(insert_dict[loop_end + 1])
        
        # check
        if all(line[0] == ' ' for line in diff_lines):
            return ''

        # remove unchanged blank lines in the beginning and end
        if any(line[1:].strip() for line in diff_lines):
            # it is not all blank lines
            start_lineno = 0
            while start_lineno < len(diff_lines) and not diff_lines[start_lineno].strip():
                start_lineno += 1
            if start_lineno > 0:
                diff_lines = diff_lines[start_lineno:]

            end_lineno = len(diff_lines) - 1
            while end_lineno >= 0 and not diff_lines[end_lineno].strip():
                end_lineno -= 1
            if end_lineno < len(diff_lines) - 1:
                diff_lines = diff_lines[:end_lineno + 1]

        return self.SPLIT_HEADER + ''.join(diff_lines)


    def _generate_search_replace_contentdiff(self, anchor_range, pos_blocks):
        '''
        Generate search-replace content diff: search/replace
        '''
        insert_dict, del_positions, loop_start, loop_end = self._preprocess_pos_blocks(anchor_range, pos_blocks)
        # remove +
        insert_dict = {k: [line[1:] for line in v] for k, v in insert_dict.items()}
        original_block = [self.source_lines[i] for i in range(loop_start, loop_end + 1)]
        
        new_block = []
        for i in range(loop_start, loop_end + 1):
            if i in insert_dict:
                new_block.extend(insert_dict[i])
            if i not in del_positions:
                new_block.append(self.source_lines[i])

        if loop_end + 1 in insert_dict:
            new_block.extend(insert_dict[loop_end + 1])
        
        original_block, new_block = self._trim_common_blank_lines(original_block, new_block, 0)

        # check
        if original_block == new_block:
            return ''

        return self.search_splitter + ''.join(original_block) + self.seperator + ''.join(new_block) + self.replace_splitter


    def _get_op_details(self, pos_blocks: list) -> Tuple[Dict, Set]:
        """ extract insert_dict and del_positions from pos_blocks"""
        insert_dict, del_positions = {}, set()
        if not pos_blocks: 
            return insert_dict, del_positions
        for numbers, content in pos_blocks:
            del_pos, del_count, ins_pos, ins_count = numbers
            if del_count > 0:
                del_positions.update(range(del_pos - 1, del_pos - 1 + del_count))
            if ins_count > 0:
                i_pos = ins_pos - 1
                add_content = [x for x in content if x.startswith('+')]
                if i_pos in insert_dict:
                    assert False, 'Conflict in Unified Diff'
                insert_dict[i_pos] = add_content
        return insert_dict, del_positions


    def _preprocess_pos_blocks(self, anchor_range: Tuple[int, int], pos_blocks: list) -> Tuple[Dict, Set, int, int]:
        """ extend anchor range to cover all operations """
        start, end = anchor_range
        insert_dict, del_positions = self._get_op_details(pos_blocks)
        
        all_op_indices = del_positions.union(insert_dict.keys())
        if all_op_indices:
            loop_start = min(start, min(all_op_indices))
            loop_end = max(end, max(all_op_indices))
        else:
            loop_start, loop_end = start, end
        
        # Clamp to valid range
        loop_start = max(0, loop_start)
        loop_end = min(len(self.source_lines) - 1, loop_end)
        
        return insert_dict, del_positions, loop_start, loop_end


    def _trim_common_blank_lines(self, block1: list, block2: list, content_index=1) -> tuple:
        """remove common blank lines"""
        if all(not line[content_index:].strip() for line in block1):
            # block1 is all blank lines
            return block1, block2

        if not block1 or not block2:
            return block1, block2

        # head
        head_trim = 0
        while (head_trim < len(block1) and head_trim < len(block2) and
               not block1[head_trim][content_index:].strip() and not block2[head_trim][content_index:].strip()):
            head_trim += 1
        if head_trim > 0:
            block1, block2 = block1[head_trim:], block2[head_trim:]

        # tail
        tail_trim = 0
        while (tail_trim < len(block1) and tail_trim < len(block2) and
               not block1[-(tail_trim + 1)][content_index:].strip() and not block2[-(tail_trim + 1)][content_index:].strip()):
            tail_trim += 1
        if tail_trim > 0:
            block1, block2 = block1[:-tail_trim], block2[:-tail_trim]
            
        return block1, block2

    # -------------------------------------------------------- #
        
    def apply_diff(self, text: str, diff: str) -> str:
        '''
        Apply the diff to the text
        '''
        text = self._ensure_newline(text)
        if not diff.strip():
            return text
        
        diff = '\n' + diff.lstrip()
        diff = self._ensure_newline(diff)

        if self.diff_shape == 'search_replace':
            # transform to unidiff-like format
            matches = self.pattern.findall(diff)
            extracted_pairs = []
            for search_content_raw, replace_content_raw in matches:
                # remove the first \n
                extracted_pairs.append((search_content_raw[1:], replace_content_raw[1:]))
            
            diff = '\n'
            for search_content, replace_content in extracted_pairs:
                delete_lines = ['-' + line for line in search_content.splitlines(keepends=True)]
                insert_lines = ['+' + line for line in replace_content.splitlines(keepends=True)]
                diff += self.SPLIT_HEADER + ''.join(delete_lines) + ''.join(insert_lines)

        return self.apply_unidiff_like_contentdiff(text, diff)

    
    def apply_unidiff_like_contentdiff(self, text: str, diff: str) -> str:
        '''
        where the content is like unified diff
        '''
        # prepare original text and diff
        split_info = diff.split('\n' + self.SPLIT_HEADER)
        diff_blocks = split_info[1:]
        for i in range(len(diff_blocks) - 1):
            # Add the splited \n back except for the last block
            diff_blocks[i] = diff_blocks[i] + '\n'

        # retain valid blocks
        valid_blocks = []
        block_set = set()
        for block in diff_blocks:
            # filter out invalid lines
            lines = [line for line in block.splitlines(keepends=True) if line[0] in {'-', '+', ' '}]
            # check if the block has valid lines (delete or insert)
            if any(line[0] in {'-', '+'} for line in lines):
                block_str = ''.join(lines)
                if block_str not in block_set:
                    block_set.add(block_str)
                    valid_blocks.append(lines)
        
        if not valid_blocks:
            return text

        original_lines = text.splitlines(keepends=True)

        # locate blocks by anchor: with tolerance
        located_blocks = []
        for diff_lines in valid_blocks:
            try:
                final_exception = None
                for current_level in range(6):
                    try:
                        # locate
                        block_info = self.find_match_blocks(original_lines, diff_lines, current_level)
                        located_blocks.append(block_info)
                        break
                    except Exception as e:
                        final_exception = e
                else:
                    raise final_exception
            except:
                if self.strict_mode:
                    raise
        
        if len(located_blocks) == 0:
            return text
        
        # merge blocks
        merged_blocks = self.merge_matched_blocks(located_blocks)

        # apply the diff
        merged_blocks.sort(key=lambda x: x[0], reverse=True)  # sort by the start position
        for anchor_range, anchor_lines in merged_blocks:
            # replace the original lines with the diff lines
            new_block = [x[1:] for x in anchor_lines if not x.startswith('-')]
            original_lines[anchor_range[0]:anchor_range[1]] = new_block
        
        # join the lines
        return ''.join(original_lines)
    

    def remove_blank_characters(self, text, current_level):
        '''
        Remove blank characters based on the match level
        '''
        if current_level == 0:
            # exact match
            return text
        elif current_level == 1:
            # remove leading and trailing blank characters
            return text.strip()
        elif current_level == 2:
            # remove all blank characters
            return ''.join(text.split())
        else:
            assert False, f"Invalid match level: {current_level}"
    

    def get_indent(self, line: str):
        index = len(line) - len(line.lstrip(' \t'))
        indent_str = line[:index]
        num = 0
        for char in indent_str:
            if char == ' ':
                num += 1
            elif char == '\t':
                num += 4
        return num


    def find_match_blocks(self, src_lines, diff_lines, current_level) -> tuple:
        '''
        0: exact match
        1: remove leading and trailing blank characters
        2: remove all blank characters
        3: ignore blank lines + 1
        4: ignore blank lines + 2
        5: ignore blank lines + 3
        '''
        if current_level >= 3:
            # src lines
            src_mapping = {}
            effective_src_lines = []
            for i, line in enumerate(src_lines):
                if line.strip():
                    src_mapping[len(effective_src_lines)] = i
                    effective_src_lines.append(line)
            if not src_mapping:
                src_mapping = {0: 0}

            # diff lines
            diff_mapping = {}
            effective_diff_lines = []
            anchor_index = 0
            for i, line in enumerate(diff_lines):
                if line.startswith('+'):
                    effective_diff_lines.append(line)
                elif line[1:].strip():
                    # effective anchor line
                    effective_diff_lines.append(line)
                    diff_mapping[anchor_index] = i
                    anchor_index += 1
            if not diff_mapping:
                diff_mapping = {0: 0}

            if effective_src_lines:
                match_block = self.find_match_blocks(effective_src_lines, effective_diff_lines, current_level - 3)

                anchor_range = match_block[0]
                # handle edge cases
                if anchor_range == (0, 0):
                    return match_block
            else:
                # insert at the beginning by default
                anchor_range = (0, 1)
            
            # update the diff lines to handle indent or dedent
            effect_index = 0
            for i, line in enumerate(diff_lines):
                if line.startswith('+') or line[1:].strip():
                    diff_lines[i] = effective_diff_lines[effect_index]
                    effect_index += 1

            start_pos, end_pos = anchor_range
            # may be adjusted by the blank lines
            actual_start_pos = src_mapping[start_pos]
            actual_end_pos = src_mapping[end_pos - 1] + 1
            # operate on the diff lines
            del_positions = []
            insert_positions = {}

            # handle blank lines
            def process_blank_lines_in_range(diff_start_index, src_start_index, position_type):
                '''
                process blank lines in the given position
                position_type: head, middle, tail

                Return: the number of blank lines in src and diff lines
                '''
                assert position_type in {'head', 'middle', 'tail'}

                if position_type == 'tail':
                    # 'down'
                    scan_step = 1
                    in_diff_range = lambda idx: idx < len(diff_lines)
                    in_src_range = lambda idx: idx < len(src_lines)
                else:
                    # 'up'
                    scan_step = -1
                    in_diff_range = lambda idx: idx >= 0
                    in_src_range = lambda idx: idx >= 0

                # obtain blank lines before the first anchor line
                diff_blank_index = diff_start_index + scan_step
                unchanged_blank_list = []
                deleted_blank_list = []
                while in_diff_range(diff_blank_index):
                    if diff_lines[diff_blank_index].startswith('+'):
                        # skip
                        pass
                    elif diff_lines[diff_blank_index][1:].strip():
                        break
                    elif diff_lines[diff_blank_index].startswith('-'):
                        deleted_blank_list.append(diff_blank_index)
                    else:
                        unchanged_blank_list.append(diff_blank_index)
                    diff_blank_index += scan_step

                diff_blank_nums = len(unchanged_blank_list) + len(deleted_blank_list)
                src_blank_nums = None
                if diff_blank_nums > 0 or position_type == 'middle':
                    # obtain neighboring blank lines of src lines
                    src_blank_index = src_start_index + scan_step
                    while in_src_range(src_blank_index) and not src_lines[src_blank_index].strip():
                        src_blank_index += scan_step
                    src_blank_nums = abs(src_start_index + scan_step - src_blank_index)

                    if src_blank_nums < diff_blank_nums:
                        # remove rebudant blank lines in diff
                        excess = diff_blank_nums - src_blank_nums
                        # unchanged blank lines first
                        for idx in unchanged_blank_list:
                            if excess <= 0:
                                break
                            del_positions.append(idx)
                            excess -= 1
                        # fix too much deletion of blank lines
                        for idx in deleted_blank_list:
                            if excess <= 0:
                                break
                            del_positions.append(idx)
                            excess -= 1

                    elif src_blank_nums > diff_blank_nums:
                        # remove rebudant blank lines in src: for head and tail, only adjust actual_xxx_pos
                        if position_type == 'middle':
                            # insert blank lines as content lines
                            insert_positions[diff_index] = src_blank_nums - diff_blank_nums

                return src_blank_nums, diff_blank_nums

            # head
            src_blank_nums, diff_blank_nums = process_blank_lines_in_range(diff_mapping[0], src_mapping[start_pos], 'head')
            if diff_blank_nums > 0:
                actual_start_pos -= min(src_blank_nums, diff_blank_nums)

            # middle
            for ef_src_index in range(start_pos + 1, end_pos):
                process_blank_lines_in_range(diff_mapping[ef_src_index - start_pos], src_mapping[ef_src_index], 'middle')
            
            # tail
            src_blank_nums, diff_blank_nums = process_blank_lines_in_range(diff_mapping[end_pos - 1 - start_pos], src_mapping[end_pos - 1], 'tail')
            if diff_blank_nums > 0:
                actual_end_pos += min(src_blank_nums, diff_blank_nums)
            
            # apply the operations
            new_diff_lines = []
            for i, line in enumerate(diff_lines):
                if i in insert_positions:
                    # original blank lines
                    for _ in range(insert_positions[i]):
                        new_diff_lines.append(' \n')
                if i not in del_positions:
                    new_diff_lines.append(line)
            
            return (actual_start_pos, actual_end_pos), new_diff_lines
    
        # match_level < 3: consider blank lines as normal lines
        matched_blocks = []

        src_length = len(src_lines)
        anchor_lines = [x[1:] for x in diff_lines if not x.startswith('+')]
        anchor_length = len(anchor_lines)

        # edge cases
        if src_length == 0:
            return (0, 0), [x for x in diff_lines if x.startswith('+')]
        elif anchor_length == 0:
            if len(diff_lines) > 0:
                raise NoMatchError("Anchor cannot be empty for non-empty text")
            return (0, 0), []

        # match the anchor lines in the source text
        for i in range(src_length - anchor_length + 1):
            for j in range(anchor_length):
                if self.remove_blank_characters(src_lines[i + j], current_level) != self.remove_blank_characters(anchor_lines[j], current_level):
                    break
            else:
                if len(matched_blocks) > 0:
                    raise MultipleMatchError("Found multiple anchor lines in the text")
                
                # replace content lines with the original src lines
                if current_level > 0:
                    # not exact match
                    src_index = i
                    diff_index = 0
                    indent_gap = None
                    while src_index < i + anchor_length:
                        while diff_index < len(diff_lines) and diff_lines[diff_index].startswith('+'):
                            diff_index += 1
                        if diff_index == len(diff_lines):
                            break

                        if indent_gap is None:
                            diff_content = diff_lines[diff_index][1:]
                            if diff_content.strip():
                                src_content = src_lines[src_index]
                                indent_gap = self.get_indent(src_content) - self.get_indent(diff_content)
                            
                        diff_lines[diff_index] = diff_lines[diff_index][0] + src_lines[src_index]
                        src_index += 1
                        diff_index += 1
                    
                    if indent_gap:
                        for index, item in enumerate(diff_lines):
                            if item.startswith('+'):
                                diff_indent = self.get_indent(item[1:])
                                new_indent = max(0, diff_indent + indent_gap)
                                diff_lines[index] = '+' + ' ' * new_indent + item[1:].lstrip(' \t')
                    
                matched_blocks.append(((i, i + anchor_length), diff_lines))
        
        if len(matched_blocks) == 0:
            raise NoMatchError("Cannot find anchor lines in the text")
        return matched_blocks[0]


    def merge_matched_blocks(self, matched_blocks):
        '''
        Merge the matched blocks into a single block
        [(anchor_range, content_lines), ]
        '''
        matched_blocks = [x for x in matched_blocks if x[1]]  # filter out empty blocks
        if len(matched_blocks) <= 1:
            return matched_blocks

        matched_blocks.sort(key=lambda x: x[0])  # sort by the position
        
        merged_blocks = []
        index = 0
        while index < len(matched_blocks):
            current_block = matched_blocks[index]
            current_left_pos, current_right_pos = current_block[0]
            current_content_lines = current_block[1]

            index += 1
            while index < len(matched_blocks):
                next_block = matched_blocks[index]
                
                # place in one state
                next_left_pos, next_right_pos = next_block[0]
                next_content_lines = next_block[1]
                # Check if the next block can be merged with the current block
                if current_left_pos == 0 and current_right_pos == 0 and next_left_pos == 0 and next_right_pos == 0:
                    # conflicts
                    raise ContentDiffError("Conflict in the same block")

                elif self.is_overlapping(current_left_pos, current_right_pos-1, next_left_pos, next_right_pos-1, 0):
                    # merge content lines
                    overlapped_left_pos = next_left_pos
                    overlapped_right_pos = min(current_right_pos, next_right_pos)
                    
                    # left
                    content_lines = []
                    left_length = overlapped_left_pos - current_left_pos
                    left_index = 0
                    while left_length > 0:
                        line = current_content_lines[left_index]
                        content_lines.append(line)
                        if line.startswith(' ') or line.startswith('-'):
                            left_length -= 1
                        left_index += 1
                    
                    # check conflicts in the overlapped region
                    overlapped_length = overlapped_right_pos - overlapped_left_pos
                    right_index = 0
                    while overlapped_length > 0:
                        # move to next anchor line
                        left_insert_count = 0
                        while current_content_lines[left_index].startswith('+'):
                            left_insert_count += 1
                            left_index += 1
                        right_insert_count = 0
                        while next_content_lines[right_index].startswith('+'):
                            right_insert_count += 1
                            right_index += 1
                        if left_insert_count > 0 and right_insert_count > 0:
                            # conflict
                            raise ContentDiffError("Conflict in the overlapped region")
                        elif left_insert_count > 0:
                            # only left has insertions, add them
                            content_lines.extend(current_content_lines[left_index-left_insert_count:left_index])
                        elif right_insert_count > 0:
                            # only right has insertions, add them
                            content_lines.extend(next_content_lines[right_index-right_insert_count:right_index])
                        else:
                            # no insertions
                            pass

                        # set delete if any
                        if current_content_lines[left_index].startswith('-'):
                            content_lines.append(current_content_lines[left_index])
                        else:
                            content_lines.append(next_content_lines[right_index])
                        
                        overlapped_length -= 1
                        left_index += 1
                        right_index += 1
                    
                    # check conflict for the end of overlapping
                    current_total_lines = len(current_content_lines)
                    next_total_lines = len(next_content_lines)
                    
                    # move to next anchor line
                    left_insert_count = 0
                    while left_index < current_total_lines and current_content_lines[left_index].startswith('+'):
                        left_insert_count += 1
                        left_index += 1
                    right_insert_count = 0
                    while right_index < next_total_lines and next_content_lines[right_index].startswith('+'):
                        right_insert_count += 1
                        right_index += 1
                    if left_insert_count > 0 and right_insert_count > 0:
                        # conflict
                        raise ContentDiffError("Conflict in the overlapped region")
                    elif left_insert_count > 0:
                        # only left has insertions, add them
                        content_lines.extend(current_content_lines[left_index-left_insert_count:left_index])
                    elif right_insert_count > 0:
                        # only right has insertions, add them
                        content_lines.extend(next_content_lines[right_index-right_insert_count:right_index])
                    else:
                        # no insertions
                        pass
                    
                    # add the remaining lines
                    if left_index < current_total_lines and right_index < next_total_lines:
                        # unexpected cases
                        raise ContentDiffError("Unexpected diff lines in the content")
                    elif left_index < current_total_lines:
                        content_lines.extend(current_content_lines[left_index:])
                    elif right_index < next_total_lines:
                        content_lines.extend(next_content_lines[right_index:])

                    # update the current block
                    current_right_pos = max(current_right_pos, next_right_pos)
                    current_content_lines = content_lines

                    index += 1
                else:
                    # end this block
                    break

            # Add the merged block to the new list
            merged_blocks.append(((current_left_pos, current_right_pos), current_content_lines))
        
        return merged_blocks
