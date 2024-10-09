import pandas as pd

# CSVファイルを読み込む
file_path = "/hpc_share/ee217092/poisoning-gradient-matching/tables/table.py"
df = pd.read_csv(file_path, delimiter="\t", on_bad_lines="skip")

# 各行のカラム数が一致しているか確認
expected_columns = len(df.columns)
for index, row in df.iterrows():
    if len(row) != expected_columns:
        print(f"Row {index} is malformed: {row}")

# 不正な行を修正する（例: 手動で修正）
# ここでは、手動で修正する例を示しますが、必要に応じてプログラムで修正することもできます。
# 例えば、以下のように不正な行を削除することができます。
df = df[df.apply(lambda x: len(x) == expected_columns, axis=1)]

# 空白セルを削除する
df = df.applymap(
    lambda x: x.strip() if isinstance(x, str) else x
)  # 文字列の前後の空白を削除
df.replace("", pd.NA, inplace=True)  # 空文字をNAに置換

# 修正されたデータフレームを保存
df.to_csv("fixed_table_ResNet18_single-class.csv", index=False, sep="\t")
