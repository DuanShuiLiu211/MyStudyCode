import sys


class Solution:
    def hashmap_set(self, strings_1, strings_2):
        strings_1 = set(list(strings_1))
        strings_2 = set(list(strings_2))

        return (strings_1 ^ strings_2).pop()


get_strings = list(map(str, sys.stdin.readline().strip().split(" ")))
result = Solution().hashmap_set(get_strings[0], get_strings[1])
sys.stdout.write(f"{result}\n")
