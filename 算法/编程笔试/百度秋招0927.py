import collections


class Solution():

    def count_diff_k(self, list_n, k):
        count, result = collections.defaultdict(int), 0
        for num in list_n:
            count[num], result = count[num] + 1, result + count[
                num + k] + count[num - k]

        return result


get_strings = []
get_strings.append(list(map(int, input().strip().split(" "))))
get_strings.append(list(map(int, input().strip().split(" "))))
k = get_strings[0][1]
list_n = get_strings[1]
result = Solution().count_diff_k(list_n, k)
print(result)


class Solution():

    def mountain(self, k, m_list, a_list, result=0):
        if k < m_list[0]:
            if result == 0:
                result = -1
            return result
        else:
            for idx in range(len(m_list)):
                if k >= max(m_list):
                    return result
                else:
                    k += a_list[idx]
                    result += 1
                    return self.mountain(k,
                                         m_list[idx + 1:],
                                         a_list[idx + 1:],
                                         result=result)

    def main(self, k, m_list, a_list, result=0):
        ans = []
        for idx in range(len(m_list)):
            pass


# 5 3
# 3 2 5 4 6
# 1 2 1 3 0

# 2

get_strings = []
get_strings.append(list(map(int, input().strip().split(" "))))
get_strings.append(list(map(int, input().strip().split(" "))))
get_strings.append(list(map(int, input().strip().split(" "))))
k = get_strings[0][1]
m_list = get_strings[1]
a_list = get_strings[2]
result = Solution().mountain(k, m_list, a_list)
print(result)
