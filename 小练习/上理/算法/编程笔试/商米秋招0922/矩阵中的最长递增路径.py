from functools import lru_cache 

class Solution:
    # 使用记忆化搜索与动态规划，状态转移公式为dp[i][j]=max(dp[i+1][j],dp[i−1][j],dp[i][j+1],dp[i][j−1],0)+1
    # 其中要求[i,j]周围的数比它小，如果没有比它小的，则长度为1
    def longestIncreasingPath(self, matrix):
        @lru_cache(None)
        def dfs(x, y):
            res = 1
            for a, b in [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]:
                if 0 <= a < m and 0 <= b < n and matrix[a][b] > matrix[x][y] and dfs(a, b) + 1 > res:
                    res = dfs(a, b) + 1
            return res

        m, n = len(matrix), len(matrix[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if dfs(i, j) > ans:
                    ans = dfs(i, j)
        return ans


matrix = [[9, 9, 4], [6, 6, 8], [2, 1, 1]]
result = Solution().longestIncreasingPath(matrix)
print(result)
