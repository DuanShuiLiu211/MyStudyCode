class Solution:
    """
    数组中取三个元素组成三角形，返回三角形的最大周长
    是一个排序问题
    """

    def merge(self, left, right):
        result = []
        while left and right:
            if left[0] <= right[0]:
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        while left:
            result.append(left.pop(0))
        while right:
            result.append(right.pop(0))
        return result

    def merge_sort(self, List):
        n = len(List)
        if n < 2:
            return List
        middle = n // 2
        left, right = List[0:middle], List[middle:]
        return self.merge(self.merge_sort(left), self.merge_sort(right))

    def subject01(self, arr):
        n = len(arr)
        if n < 3:
            return 0
        arr = self.merge_sort(arr)
        arr_list = []
        for i in range(n - 3, 0, -1):
            if arr[i] < (arr[i + 1] + arr[i + 2]):
                arr_list.append(arr[i] + arr[i + 1] + arr[i + 2])
                break

        if len(arr_list) == 0:
            return 0

        return max(arr_list)


if __name__ == "__main__":
    a = [3, 5, 8, 10, 4]
    b = Solution()
    c = b.subject01(a)
    print(c)
