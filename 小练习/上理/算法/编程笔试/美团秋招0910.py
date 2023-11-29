class Solution:
    @staticmethod
    def subject_1(t, lists):
        assert 1 <= t <= 1000
        ans_list = []
        for idx in range(len(lists)):
            n = lists[idx][0]
            x = lists[idx][1]
            y = lists[idx][2]
            k = lists[idx][3]
            if (n - k + 1) / x == (k / y):
                ans_list.append("Tie")
            elif (n - k + 1) / x > (k / y):
                ans_list.append("Win")
            else:
                ans_list.append("Lose")

        return ans_list


get_strings = [int(input())]
t_lists = []
i = 1
while i <= get_strings[0]:
    t_lists.append(list(map(int, input().strip().split(" "))))
    i += 1
get_strings.append(t_lists)

ans_list = Solution().subject_1(get_strings[0], get_strings[1])
for ans in ans_list:
    print(ans)


class Solution:
    @staticmethod
    def subject_2(n, lists):
        k = 0
        ans = 1
        for idx in range(n):
            ans *= lists[idx]
        if sum(lists) != 0 and ans != 0:
            return k
        else:
            if ans == 0:
                zeros = 0
                for idx in range(n):
                    if lists[idx] == 0:
                        zeros += 1
                if sum(lists) != -zeros:
                    k = zeros
                else:
                    if min(lists + 1) != 0:
                        k = zeros + 1
            else:
                zeros = 0
                for idx in range(n):
                    if lists[idx] == 0:
                        zeros += 1
                if sum(lists) != -zeros:
                    k = zeros
                else:
                    if min(lists + 1) == 0:
                        k = zeros + 1
                    else:
                        k = 1
                return k


get_strings = [int(input()), list(map(int, input().strip().split(" ")))]
k = Solution().subject_2(get_strings[0], get_strings[1])
print(k)


class Solution:
    @staticmethod
    def subject_3(n, lists_1, lists_2):
        if len(lists_2) <= 2:
            return max(lists_2)
        dp = [[0, 0]] * (n + 1)
        dp[0] = [0, 0]
        dp[1] = [0, lists_2[0]]
        dp[2] = [0, lists_2[1]]
        for i in range(2, n):
            if dp[i // 2][1] > dp[(i - 1) // 2][1]:
                dp[i][0] = i // 2
            else:
                dp[i][1] = dp[(i - 1) // 2][1] + lists_2[i]
                dp[i][0] = (i - 1) // 2
        ans = 0
        for i in range(len(dp)):
            if dp[i][1] > ans:
                ans = dp[i][1]
        return ans


get_strings = [
    int(input()),
    list(map(int, input().strip().split(" "))),
    list(map(int, input().strip().split(" "))),
]

k = Solution().subject_3(get_strings[0], get_strings[1], get_strings[2])
print(k)

print("运行结束")
