import heapq
import sys


# ACM模式
class Solution:
    def max_sliding_windows(self, nums, k):
        n = len(nums)
        q = [(-nums[i], i) for i in range(k)]
        heapq.heapify(q)
        ans = [-q[0][0]]
        for i in range(k, n):
            heapq.heappush(q, (-nums[i], i))
            while q[0][1] <= i - k:
                heapq.heappop(q)
            ans.append(-q[0][0])

        return ans


# inputs:[2,3,4,2,6,2,5,1],3 -> outputs:[4, 4, 6, 6, 6, 5]
lists = list(
    map(
        int,
        sys.stdin.readline().strip().replace(",", "").replace("[", "").replace("]", ""),
    )
)
nums = lists[:-1]
k = lists[-1]
ans = Solution().max_sliding_windows(nums, k)
print(ans)
