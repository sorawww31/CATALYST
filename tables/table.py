import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルの読み込み
csv_file = "tables/wandb_folder/wandb_export_2024-12-27T19_14_31.212+09_00.csv"
data = pd.read_csv(csv_file)

# 列名の確認
print(data.columns)

# データの抽出
epochs = range(len(data))
average_cosine_similarity = data["2111111100_2024-12-27 - average_cosine_similarity "]
average_cosine_similarity_min = data[
    "2111111100_2024-12-27 - average_cosine_similarity__MIN "
]
average_cosine_similarity_max = data[
    "2111111100_2024-12-27 - average_cosine_similarity__MAX"
]

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(
    epochs, average_cosine_similarity, label="Average Cosine Similarity", color="dodb"
)
plt.fill_between(
    epochs,
    average_cosine_similarity_min,
    average_cosine_similarity_max,
    color="b",
    alpha=0.2,
)

# グラフの装飾
plt.title("Average Cosine Similarity over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Similarity")
plt.legend()
plt.grid(True)

# グラフの保存
plt.savefig("average_cosine_similarity.png")

# グラフの表示
plt.show()
