import collections
import sys


class Solution:
    def min_operations(self, nums, x):
        sum_nums = sum(nums)
        if sum_nums == x:
            return len(nums)
        last = sum_nums - x
        pre = [0]
        for i in range(len(nums)):
            pre.append(pre[-1] + nums[i])
        res = float("-inf")
        dic = collections.defaultdict(int)
        for index, value in enumerate(pre):
            if value - last in dic:
                res = max(res, index - dic[value - last])
            if value not in dic:
                dic[value] = index
        return -1 if res == float("-inf") else len(nums) - res


get_strings = []
get_strings.append(
    list(map(int, sys.stdin.readline().replace("[", "").replace("]", "").split(",")))
)
get_strings.append(int(sys.stdin.readline().strip()))
get_strings.append(int(sys.stdin.readline().strip()))


result = Solution().min_operations(get_strings[0], get_strings[-1])
print(result)
