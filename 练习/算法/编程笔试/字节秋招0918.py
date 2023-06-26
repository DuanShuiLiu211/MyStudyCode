import time
import sys


class Solution:
    def subject2(self, n_layer, n_place):
        enable_place = sorted(n_place[0][1:])
        for k in range(1, n_layer):
            for idx in range(n_place[k][0]):
                for i in range(len(enable_place)):
                    if n_place[k][idx+1] < enable_place[i]+50:
                        enable_place.append(n_place[k][idx + 1])
                        enable_place = sorted(enable_place)
                        break
                    if (i < len(enable_place)-1 and
                        n_place[k][idx+1] + 100 > enable_place[i+1] and
                        n_place[k][idx+1] < enable_place[i] + 100):
                        enable_place.append(n_place[k][idx+1])
                        enable_place = sorted(enable_place)
                        break

        return len(enable_place)


get_strings = []
get_strings.append(int(sys.stdin.readline().strip()))
nums = []
for _ in range(get_strings[0]):
    nums.append(list(map(int, sys.stdin.readline().strip().split(' '))))
get_strings.append(nums)

start_time = time.time()
result = Solution().subject2(get_strings[0], get_strings[1])
end_time = time.time()

sys.stdout.write(f"{result}\n")
print(f"{(end_time-start_time)*1000:.4f}ms")
