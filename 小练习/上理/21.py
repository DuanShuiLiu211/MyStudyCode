import re

a_string = (
    "大佬们这个怎么用正则删。 //@张三:各位同学大家好！因课题组实验需要，现征集20电动车用于视频检测实验。"
    "电动车仅用于录制一段视频，时间约10分钟，不出校门。 //@李四:结束后，给予每车30元补助。有意者可联系我。"
)

q1 = r"(?<= ).+?(?=:)"
q2 = r"[:,：]"
patten1 = re.compile(q1)
patten2 = re.compile(q2)
print(patten1.findall(a_string))
print(patten2.findall(a_string))
temp = re.sub(patten1, "", a_string)
result = re.sub(patten2, "", temp)
print("正则化命名后", result)
