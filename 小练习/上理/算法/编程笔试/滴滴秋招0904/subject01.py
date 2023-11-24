class Solution:
    def get_subsets(self, string):
        # 全部子集
        result = [[]]
        size = len(string)
        for i in range(size):
            for j in range(len(result)):
                # 现有的每个子集中都添加新元素，作为新子集加入结果集中
                result.append(result[j] + [string[i]])

        return result

    def subject01(self, samples) -> int:
        n = samples[0][0]
        k = samples[0][1]
        nums = samples[1]
        subset = self.get_subsets(nums)
        ans = []
        for set in subset:
            if set:
                if max(set) <= k * sum(set) / len(set):
                    ans.append(len(set))

        return max(ans)


data = [[5, 2], [3, 10, 5, 4, 2]]
print(Solution().subject01(data))
