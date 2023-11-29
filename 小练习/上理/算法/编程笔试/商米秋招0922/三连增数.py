def solution(lists):
    # 在这⾥写代码
    string_result = ""
    end = 0
    for idx in range(len(lists) - 1):
        if lists[idx] + 1 == lists[idx + 1]:
            end += 1
        else:
            if end >= 2:
                string_result += str(lists[idx - end]) + "-" + str(lists[idx]) + ","
                end = 0
            else:
                string_result += str(lists[idx - end : idx + 1])[1:-1] + ","
                end = 0

    return string_result[: len(lists) + 1]
