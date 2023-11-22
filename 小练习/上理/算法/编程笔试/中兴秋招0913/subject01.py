import sys


class Solution:
    def bottom_up_coins(self, row_coins):
        dp = [None] * (len(row_coins) + 1)
        dp[0] = 0
        dp[1] = row_coins[0]

        for i in range(2, len(row_coins) + 1):
            dp[i] = max(dp[i - 2] + row_coins[i - 1], dp[i - 1])

        return dp

    # 通过对 dp 的回溯来反向查找被选择的硬币
    def trace_back_coins(self, c, dp):
        select = []
        i = len(c)
        while i >= 1:
            if dp[i] > dp[i - 1]:
                select.append(c[i - 1])
                i -= 2
            else:
                i -= 1

        return sum(select)


get_strings = list(map(int, sys.stdin.readline().strip().split(" ")))
dp = Solution().bottom_up_coins(get_strings)
result = Solution().trace_back_coins(get_strings, dp)
sys.stdout.write(f"{result}\n")
