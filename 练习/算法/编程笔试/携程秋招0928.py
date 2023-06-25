class Solution():

    def mean_diff_max(self, n, k, list_n):
        if k == 1:
            return max(list_n) - min(list_n)
        elif k == n:
            return 0
        elif k == 2:
            list_n = sorted(list_n)
            if (list_n[0] + list_n[-1])/2 > list_n[-2]:
                return (list_n[0] + list_n[-1]) / 2 - list_n[1]
            elif (list_n[0] + list_n[-1])/2 < list_n[1]:
                return list_n[-2] - (list_n[0] + list_n[-1]) / 2
            else:
                return list_n[-2] - list_n[1]
        else:
            list_n = sorted(list_n)
            diff = []
            for idx in range(2):
                if idx == 0:
                    l = k // 2
                    r = k - l
                    lr_mean = sum(list_n[:l] + list_n[-r:])/k
                    if lr_mean > list_n[-(r+1)]:
                        diff.append(lr_mean - list_n[l])
                    elif lr_mean < list_n[l]:
                        diff.append(list_n[-(r+1)] - lr_mean)
                    else:
                        diff.append(list_n[-(r+1)] - list_n[l])
                else:
                    r = k // 2
                    l = k - r
                    rl_mean = sum(list_n[:l] + list_n[-r:])/k
                    if rl_mean > list_n[-(r+1)]:
                        diff.append(rl_mean - list_n[l])
                    elif rl_mean < list_n[l]:
                        diff.append(list_n[-(r+1)] - rl_mean)
                    else:
                        diff.append(list_n[-(r+1)] - list_n[l])

            return min(diff)


get_strings = []
get_strings.append(list(map(int, input().strip().split(" "))))
get_strings.append(list(map(int, input().strip().split(" "))))
n = get_strings[0][0]
k = get_strings[0][1]
list_n = get_strings[1]
result = Solution().mean_diff_max(n, k, list_n)
print(result)