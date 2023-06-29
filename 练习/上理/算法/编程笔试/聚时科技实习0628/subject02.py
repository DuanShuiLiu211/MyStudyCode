class Solution:
    """
    假设a是A中的第⼀个元素，要求该函数返回A的⼀个重排列B，使得a在B中左边的元素都⼩于a，右边的元素都⼤于等于a
    要求:时间复杂度为O(n),空间复杂度为O(1)
    是一个双指针问题
    """
    def subject02(self, arr):
        n = len(arr)
        if n < 2:
            return arr

        # 慢指针
        l = 0
        # 快指针
        f = 1

        # 从第一个元素开始遍历
        while f < n:
            if arr[l] > arr[f]:
                arr[f], arr[l] = arr[l], arr[f]
                if l < f-1:
                    arr[f], arr[l + 1] = arr[l + 1], arr[f]
                    l += 1
                else:
                    l = f
            else:
                f += 1
        while f < n:
            if arr[f] > arr[l]:
                f = f + 1
            else:
                arr[f], arr[l] = arr[l], arr[f]
                l = f
                f = f + 1

        f = l + 1
        while f < 0 and l >= 0:
            if arr[l] > arr[f]:
                arr[f], arr[l] = arr[l], arr[f]
                l = f
                f = l - 1
            else:
                f = f - 1

        return arr


if __name__ == '__main__':
    # 有问题
    a = [2, 3, 1, 4, 7, 8, 5, 0, -1]
    b = Solution()
    c = b.subject02(a)
    print(c)

