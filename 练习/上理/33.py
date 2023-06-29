import pandas as pd

# pandas入门
df1 = pd.DataFrame({'one': "[1, 2, 3]", "two": "[4, 5, 6]"}, index=[0])

# 1. pandas操作excel
with pd.ExcelWriter('excel_1.xlsx') as writer:
    df1.to_excel(writer,
                 sheet_name='Sheet1',
                 index=True,
                 header=True,
                 startrow=0,
                 startcol=0)
