from .contentdiff import ContentDiffTool
from .minunidiff import MinUniDiffTool


class ContentDiffError(Exception):
    pass

class MultipleMatchError(ContentDiffError):
    pass


class MinContentDiffTool(ContentDiffTool):
    def __init__(self, diff_shape='standard', strict_mode=False, min_context_lines=0):
        assert min_context_lines >= 0, "Minimum anchor lines must be non-negative."
        # merge close hunks
        super().__init__(1, diff_shape, strict_mode)
        self.min_context_lines = min_context_lines
        self.min_unidiff_tool = MinUniDiffTool(strict_mode=True)


    def calculate_diff(self, text1: str, text2: str, **kwargs) -> str:
        unidiff = self.min_unidiff_tool.calculate_diff(text1, text2)
        if not unidiff:
            return unidiff
        
        original_text = self._ensure_newline(text1)
        self.source_lines = original_text.splitlines(keepends=True)
        diff_list = self.min_unidiff_tool.parse_diff(unidiff)

        if not original_text:
            # special case: empty original text
            content_diff_blocks = [((0, -1), diff_list)]
        else:
            # original states
            self.compare_source_lines = [''.join(x.split()) for x in self.source_lines if x.strip()]

            # content-addressed hunks with minimal anchors
            content_diff_blocks = self.convert_unified_diff(diff_list)
            
            # merge diff hunks with overlapping lines
            content_diff_blocks = self.merge_diff_blocks(content_diff_blocks)

        # convert to content-addressed diff formats
        ret = self.restore_diff_text(content_diff_blocks)
        
        return ret
    

    def convert_unified_diff(self, diff_list) -> list:
        '''
        Convert unified diff to content-addressed diff hunks (one-to-one mapping)
        '''
        content_diff_blocks = []
        offset = 0
        for (numbers, content) in diff_list:
            # each diff block
            del_pos, del_count, ins_pos, ins_count = numbers
            # adjust the position based on the original text
            ins_pos -= offset

            anchor_range = self.find_minimal_anchor(del_pos, del_count, ins_pos)
            content_diff_blocks.append((anchor_range, [((del_pos, del_count, ins_pos, ins_count), content)]))

            offset += ins_count - del_count

        return content_diff_blocks


    def find_minimal_anchor(self, del_pos, del_count, ins_pos):
        '''
        Find the minimal anchor for the given deletion and insertion positions
        '''
        if del_count > 0:
            start_pos = del_pos - 1  # Convert to zero-based index
            end_pos = start_pos + del_count - 1
        else:
            # only insertions
            start_pos = ins_pos - 1 - 1  # Convert to zero-based index and move to the line before insertion
            end_pos = start_pos + 1      # the line after insertion
        
        total_lines = len(self.source_lines)
        max_inc = max(0, start_pos, total_lines - 1 - end_pos)    # maximum anchor lines to use
        # in the worst case, it can use all lines to locate
        for i in range(0, max_inc + 1):
            left_pos = max(0, start_pos - i)
            right_pos = min(total_lines - 1, end_pos + i)
            if self.is_content_unique(left_pos, right_pos):
                # Found a unique anchor
                if i < self.min_context_lines:
                    # Extend the anchor to meet the minimum context lines requirement
                    left_pos = max(0, start_pos - self.min_context_lines)
                    right_pos = min(total_lines - 1, end_pos + self.min_context_lines)
                    
                return left_pos, right_pos
        
        # Fallback to the whole text if no unique anchor found
        return 0, total_lines - 1

    
    def merge_diff_blocks(self, diff_blocks):
        '''
        [(anchor_range, [((del_pos, del_count, ins_pos, ins_count), content)])]
        '''
        if len(diff_blocks) <= 1:
            return diff_blocks

        new_blocks = []

        index = 0
        while index < len(diff_blocks):
            current_block = diff_blocks[index]
            current_left_pos, current_right_pos = current_block[0]

            start_index = index

            index += 1
            while index < len(diff_blocks):
                next_block = diff_blocks[index]
                
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
            end_index = index
            pos_blocks = []
            for x in diff_blocks[start_index:end_index]:
                pos_blocks.extend(x[1])
            new_blocks.append(((current_left_pos, current_right_pos), pos_blocks))
        
        return new_blocks


if __name__ == "__main__":
    from example import source_code, target_code

    diff_tool = MinContentDiffTool("standard", True, 3)
    diff = diff_tool.calculate_diff(source_code, target_code)
    print(diff)

    patched_code = diff_tool.apply_diff(source_code, diff)
    print(patched_code)

    assert diff_tool._ensure_newline(patched_code) == diff_tool._ensure_newline(target_code), "The patched code does not match the target code."
