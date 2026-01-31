"""Format checking utilities for ReForm output validation."""

import re


def check_answer_part(text: str) -> bool:
    """
    Check if the answer part contains any forbidden tags.
    
    Args:
        text: The answer text to check.
        
    Returns:
        True if the answer is valid (no forbidden tags), False otherwise.
    """
    forbidden_tags = ["<think>", "</think>", "<round>", "</round>"]
    for tag in forbidden_tags:
        if tag in text:
            print(f"Error: {tag} should not appear in the answer, {text=}")
            return False
    return True


def check_output_format(text: str) -> bool:
    """
    Check if <round> and </round> tags are properly matched, closed, and not nested.
    
    Args:
        text: The text to check.
        
    Returns:
        True if the format is valid, False otherwise.
    """
    # Find all tag positions
    open_positions = [m.start() for m in re.finditer(r'<round>', text)]
    close_positions = [m.start() for m in re.finditer(r'</round>', text)]
    
    if len(open_positions) == 0 or len(close_positions) == 0:
        print(f"Error: <round> or </round> count is zero, {open_positions=}, {close_positions=}")
        return False

    # Check if counts match
    if len(open_positions) != len(close_positions):
        print(f"Error: Opening tags ({len(open_positions)}) and closing tags ({len(close_positions)}) count mismatch")
        return False
    
    # Sort all tags by position
    all_tags = [(pos, 'open') for pos in open_positions] + [(pos, 'close') for pos in close_positions]
    all_tags.sort()
    
    # Use stack to check pairing and nesting
    stack = []
    for pos, tag_type in all_tags:
        if tag_type == 'open':
            if stack:  # Stack not empty means nesting detected
                print(f"Error: Nested <round> tag detected at position {pos}")
                return False
            stack.append(pos)
        else:  # close tag
            if not stack:
                print(f"Error: </round> at position {pos} has no matching <round>")
                return False
            stack.pop()
    
    # Check for unclosed tags
    if stack:
        print(f"Error: {len(stack)} <round> tag(s) are unclosed")
        return False
    
    return True


if __name__ == "__main__":
    test_cases = [
        '<round>content1</round>',  # True
        '<round>content1</round>\ndfasf<round>content2</round><round>content2</round>',  # True
        'just text',  # False
        '<round>content1',  # False
        'content1</round>',  # False
        '<round>outer<round>inner</round>outer</round>',  # False
        '',
        '<round>content1</round><round>content2'
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {check_output_format(text)}")
