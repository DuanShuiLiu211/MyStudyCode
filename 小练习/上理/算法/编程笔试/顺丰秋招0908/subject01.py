import sys


class Solution:
    # 全部连续子串
    def get_all_series_subsets(self, string):
        length = len(string)
        lists = []
        for i in range(length):
            for j in range(i, length):
                lists.append(string[i : j + 1])
        return lists

    def subject01(self, lists):
        n = lists[0][0]
        nums = lists[1]
        subset = self.get_all_series_subsets(nums)
        max = 0
        for list in subset:
            if list:
                temp = 2 * sum(list) - len(list)
                if temp > max:
                    max = temp

        return max


class Solution:
    def subject01(self, lists):
        string = lists[1]
        length = len(string)
        max = 0
        for i in range(length):
            for j in range(i, length):
                temp = 2 * sum(string[i : j + 1]) - len(string[i : j + 1])
                if temp > max:
                    max = temp
        return max


lists = []
lists.append(list(map(int, sys.stdin.readline().strip().split())))
lists.append(list(map(int, list(sys.stdin.readline().strip()))))

print(Solution().subject01(lists))
