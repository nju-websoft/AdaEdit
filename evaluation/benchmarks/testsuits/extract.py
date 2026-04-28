import ast

def syntax_check(code: str) -> bool:
    # only support Python 3
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False

def code_extract(text: str) -> str:
    '''
    Get the longest valid code block from the text: refer to evalplus
    '''
    lines = text.splitlines(keepends=True)
    line_length = len(lines)

    # calculate prefix sum of non-empty lines
    non_empty_counts = [1 if line.strip() else 0 for line in lines]
    prefix_sum = [0]
    for count in non_empty_counts:
        prefix_sum.append(prefix_sum[-1] + count)
    
    def get_non_empty_count(start: int, end: int) -> int:
        return prefix_sum[end + 1] - prefix_sum[start]

    longest_line_pair = (0, 0)
    longest_so_far = 0

    line_length = len(lines)
    for i in range(line_length):
        if get_non_empty_count(i, line_length - 1) <= longest_so_far:
            break
        
        current_lines = lines[i]
        for j in range(i + 1, line_length):
            current_length = get_non_empty_count(i, j)
            if current_length <= longest_so_far:
                # skip if the current length is not longer than the longest found so far
                continue

            current_lines += lines[j]
            if syntax_check(current_lines):
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])
