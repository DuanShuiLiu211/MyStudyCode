class Solution:
    def get_subset(self, array):
        """
        :type originArray:list
        :rtype :listlist
        """
        result = [[]]
        size = len(array)
        for i in range(size):
            for j in range(len(result)):
                # 现有每个子集中添加新元素，作为新子集加入结果集中
                result.append(result[j]+[array[i]])

        return result

    def subject01(self, samples) -> int:
        n = samples[0][0]
        k = samples[0][1]
        nums = samples[1]
        subset = self.get_subset(nums)
        ans = []
        for set in subset:
            if set:
                if max(set) <= k * sum(set) / len(set):
                    ans.append(len(set))

        return max(ans)


data = [[5, 2], [3, 10, 5, 4, 2]]
print(Solution().subject01(data))