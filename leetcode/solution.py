import sys

from leetcode.leet_class import ListNode


class Solution:

    def __init__(self):
        pass

    """ a zigzag pattern"""

    @staticmethod
    def convert_to_zigzag_pattern(s: str, num_rows: int) -> str:
        if num_rows == 1:
            return s
        zigzag_list = list()
        current_row_num = 0
        row_direction = False
        total_rows_num = min(num_rows, len(s))
        for char in s:
            if len(zigzag_list) < total_rows_num:
                zigzag_list.append('')
            zigzag_list[current_row_num] += char
            if current_row_num == 0 or current_row_num == total_rows_num - 1:
                row_direction = not row_direction
            current_row_num += 1 if row_direction else -1
        out_put = ''
        for out_part in zigzag_list:
            out_put += out_part
        return out_put

    """longest common prefix"""

    @staticmethod
    def longest_common_prefix(str_list: list) -> str:
        if len(str_list) == 0:
            return ''
        prefix = str_list[0]
        if isinstance(prefix, str):
            for str_element in str_list:
                if isinstance(str_element, str):
                    while str_element.find(prefix) != 0:
                        prefix = prefix[:len(prefix) - 1]
                        if len(prefix) == 0:
                            return ''
                else:
                    raise Exception('please input list of str')
            return prefix
        else:
            raise Exception('please input list of str')

    """3Sum Closest"""

    @staticmethod
    def three_sum_closest(nums: list, target: int) -> int:
        if len(nums) < 3:
            return sys.maxsize
        nums.sort()
        result = nums[0] + nums[1] + nums[-1]
        for i in range(0, len(nums) - 2):
            end_pointer = len(nums) - 1
            start_pointer = i + 1
            while start_pointer < end_pointer:
                sum_value = nums[i] + nums[start_pointer] + nums[end_pointer]
                if sum_value > target:
                    end_pointer -= 1
                else:
                    if sum_value < target:
                        start_pointer += 1
                    else:
                        return sum_value
                if abs(target - sum_value) < abs(target - result):
                    result = sum_value
        return result

    """ Letter Combinations of a Phone Number"""

    @staticmethod
    def letter_combinations(digits: str) -> list:
        map_dict = {2: 'abc', 3: 'def', 4: 'ghi', 5: 'jkl', 6: 'mno', 7: 'pqrs', 8: 'tuv', 9: 'wxyz'}
        result_list = list()
        for number_str in digits:
            try:
                number = int(number_str)
                if number not in map_dict:
                    raise Exception('请输入有效数字')
                else:
                    letters = map_dict.get(number)

                    if len(result_list) == 0:
                        for s in letters:
                            result_list.append(s)
                    else:
                        temp_list = result_list.copy()
                        result_list.clear()
                        for word in temp_list:
                            for s in letters:
                                result_list.append(word + s)
            except ValueError:
                print('请输入有效数字')

        return result_list

    """4Sum"""

    @staticmethod
    def four_sum(nums: list, target):
        def find_sum(low_pointer, high_pointer, target_n, n, result: list, result_list: list):
            if high_pointer - low_pointer + 1 < n or n < 2 or n * nums[low_pointer] > target_n \
                    or target_n > n * nums[high_pointer]:
                return
            if n == 2:
                while low_pointer < high_pointer:
                    if nums[low_pointer] + nums[high_pointer] == target_n:
                        result_list.append(result.extend([nums[low_pointer], nums[high_pointer]]))
                        low_pointer += 1
                        while low_pointer < high_pointer and nums[low_pointer] == nums[low_pointer - 1]:
                            low_pointer += 1
                    else:
                        if nums[low_pointer] + nums[high_pointer] < target_n:
                            low_pointer += 1
                        else:
                            high_pointer -= 1
            else:
                for i in range(low_pointer, high_pointer + 1):
                    if i == low_pointer or (i > low_pointer and nums[i] != nums[i - 1]):
                        find_sum(i + 1, high_pointer, target_n - nums[i], n - 1, result + [nums[i]], result_list)

        nums.sort()
        results = []
        find_sum(0, len(nums) - 1, target, 4, [], results)
        return results

    """Remove Nth Node From End of List"""

    @staticmethod
    def remove_nth_from_end(head: ListNode, n):
        temp_node = ListNode(-1)
        temp_node.next = head
        first_pointer = temp_node
        second_pointer = temp_node
        for i in range(0, n + 1):
            first_pointer = first_pointer.next
        while first_pointer is not None:
            second_pointer = second_pointer.next
            first_pointer = first_pointer.next
        second_pointer.next = second_pointer.next.next
        return temp_node.next

    """Valid Parentheses"""

    @staticmethod
    def is_valid(s: str) -> bool:
        def find_tag_for_char(char):
            return {
                '(': 1,
                ')': -1,
                '{': 2,
                '}': -2,
                '[': 3,
                ']': -3
            }.get(char, 0)

        char_list = list()
        if len(s) == 0:
            return True
        for char_str in s:
            tag = find_tag_for_char(char_str)
            if tag == 0:
                return False
            else:
                if tag > 0:
                    char_list.append(tag)
                else:
                    if len(char_list) == 0:
                        return False
                    last_tag = char_list.pop()
                    if last_tag + tag != 0:
                        return False
        return len(char_list) == 0

    """Merge two sorted list"""

    @staticmethod
    def merge_two_lists(l1: ListNode, l2: ListNode):
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        if l1.val <= l2.val:
            l1.next = Solution.merge_two_lists(l1.next, l2)
            return l1
        else:
            l2.next = Solution.merge_two_lists(l1, l2.next)
            return l2

    """Generate Parentheses"""

    @staticmethod
    def generate_parenthesis(n: int) -> list:
        def find_parenthesis(result: list, cur_str, open_num, close_num, max_num):
            if len(cur_str) >= max_num * 2:
                result.append(cur_str)
                return
            if open_num < max_num:
                find_parenthesis(result, cur_str + '(', open_num + 1, close_num, max_num)
            if close_num < open_num:
                find_parenthesis(result, cur_str + ')', open_num, close_num + 1, max_num)

        parenthesis_list = list()
        find_parenthesis(parenthesis_list, '', 0, 0, n)
        return parenthesis_list

    """Merge k Sorted Lists"""

    @staticmethod
    def merge_k_sorted_lists(lists: list) -> ListNode:
        def merge_two_sorted_lists(l1: ListNode, l2: ListNode):
            if l1 is None:
                return l2
            if l2 is None:
                return l1
            if l1.val <= l2.val:
                l1.next = merge_two_sorted_lists(l1.next, l2)
                return l1
            else:
                l2.next = merge_two_sorted_lists(l1, l2.next)
                return l2

        length = len(lists)
        interval = 1
        while interval < length:
            for i in range(0, length - interval, interval * 2):
                lists[i] = merge_two_sorted_lists(lists[i], lists[i + interval])
            interval *= 2
        return lists[0] if length > 0 else lists
