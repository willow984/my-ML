import openpyxl
import pandas as pd
import random


num_rows = 100
class_range = [0, 3]
item_level_range = [0, 3]
enchantment_gem_range = [0, 2]
score_range = [0, 4]
server_range = [0, 1]

data = {
    "class": [random.randint(*class_range) for _ in range(num_rows)],
    "item_level": [random.randint(*item_level_range) for _ in range(num_rows)],
    "enchantment_gem": [random.randint(*enchantment_gem_range) for _ in range(num_rows)],
    "score": [random.randint(*score_range) for _ in range(num_rows)],
    "server": [random.randint(*server_range) for _ in range(num_rows)],
}

df = pd.DataFrame(data)
file_path = './playerInfo.xlsx'  # 替换为您希望保存文件的路径
df.to_excel(file_path, index=False)