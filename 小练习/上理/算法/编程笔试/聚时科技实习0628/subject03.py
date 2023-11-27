class Solution:
    """
    二维子矩阵的和
    是一个递推问题
    """

    def __init__(self, matrix):
        m, n = len(matrix), (len(matrix[0]) if matrix else 0)
        self.sums = [[0] * (n + 1) for _ in range(m + 1)]
        _sums = self.sums

        for i in range(m):
            for j in range(n):
                _sums[i + 1][j + 1] = (
                    _sums[i][j + 1] + _sums[i + 1][j] - _sums[i][j] + matrix[i][j]
                )

    def subject03(self, row1, col1, row2, col2):
        _sums = self.sums

        return (
            _sums[row2 + 1][col2 + 1]
            - _sums[row1][col2 + 1]
            - _sums[row2 + 1][col1]
            + _sums[row1][col1]
        )


if __name__ == "__main__":
    a = [
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5],
    ]
    b = Solution(a)
    c = b.subject03(2, 1, 4, 3)
    print(c)
