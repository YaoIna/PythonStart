from leetcode.leet_class import ListNode
from leetcode.solution import Solution

zigzag_string = Solution.convert_to_zigzag_pattern('PAYPALISHIRING', 4)
print(zigzag_string)

print(Solution.longest_common_prefix(["flower", "flow", "flight"]))

Solution.three_sum_closest([-1, 2, 1, -4], 1)

print(Solution.letter_combinations('23'))

node_5 = ListNode(5)
node_4 = ListNode(4)
node_4.next = node_5
node_3 = ListNode(3)
node_3.next = node_4
node_2 = ListNode(2)
node_2.next = node_3
node_1 = ListNode(1)
node_1.next = node_2
mm = Solution.remove_nth_from_end(node_1, 2)
tt = 1
