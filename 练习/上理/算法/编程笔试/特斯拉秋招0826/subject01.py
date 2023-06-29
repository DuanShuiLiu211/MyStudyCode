class Solution:
    """
    求数组中不同数字的种类与各自的频数，删除一些数字使得各自频数为阶梯分布，返回需要删除的个数
    1. [1, 1, 1, 2, 2, 2] -> delete 2 -> return 1
    2. [5, 3, 3, 2, 5, 2, 3, 2] -> delete [3, 3] -> return 2
    3. [1, 2, 3, 4, 5] -> delete [1, 2, 3, 4] -> 4
    """
    def subject01(self, arr):
        unique_arr = list(set(arr))  # [1, 2] / [2, 3, 5]  # 不重复的3个数
        index_arr = [0 for _ in range(len(unique_arr))]  # [3, 3] / [3, 3, 2]  # 3个数的频数
        for num in arr:
            index = unique_arr.index(num)
            index_arr[index] += 1

        unique_index_arr = list(set(index_arr))  # [3] / [3, 2]  # 不重复的频数
        index_index_arr = [0 for _ in range(len(unique_index_arr))]  # [2] / [2, 1]  # 不重复的频数的频数
        for num in index_arr:
            index = unique_index_arr.index(num)
            index_index_arr[index] += 1

        index_enable = [i+1 for i in range(len(unique_arr))]  # [1, 2] / [1, 2, 3]
        index_num = list(set(index_enable) ^ set(index_arr))  # [1] / [1, 2]

        if len(index_num) == 0:
            return 0
        else:
            # 先看包括那些频数
            outputs = 0
            for num in index_num:
                if num not in unique_index_arr:
                    temp = unique_index_arr[index_index_arr.index(num)] - index_num.pop()
                    outputs += temp
