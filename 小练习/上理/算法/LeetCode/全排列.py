"""
题目：
给定一个不含重复数字的数组 nums，返回其所有可能的全排列。你可以按任意顺序返回答案。

示例：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

输入：nums = [0,1]
输出：[[0,1],[1,0]]

输入：nums = [1]
输出：[[1]]

分析：
回溯法：一种通过探索所有可能的候选解来找出所有的解的算法。
如果候选解被确认不是一个解（或者至少不是最后一个解），
回溯算法会通过在上一步进行一些变化抛弃该解，即回溯并且再次尝试。
DFS   
"""

import itertools
from typing import List


class Solution1:
    def permute(self, nums: List[int]):
        return list(itertools.permutations(nums))


class Solution2:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(first=0):
            # 所有数都填完了
            if first == n:
                res.append(nums[:])
                return
            for i in range(first, n):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]

        n = len(nums)
        res = []
        backtrack()
        return res


def dfs(arr, depth, result):
    if depth == len(arr):
        result.append(arr[:])
        return

    for i in range(depth, len(arr)):
        arr[i], arr[depth] = arr[depth], arr[i]
        dfs(arr, depth + 1, result)
        arr[i], arr[depth] = arr[depth], arr[i]


class Solution3:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if nums is None or len(nums) == 0:
            return []

        result = []
        dfs(nums, 0, result)
        return result


a = Solution1()
b = [3, 2, 1, 6, 8, 9, 11, 13, 15]
c = a.permute(b)
print(c)
