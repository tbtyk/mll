import os
import pandas as pd
import matplotlib.pyplot as plt
import wandb  # 追加: Weights & Biases (学習記録の保存用)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
# 1. データの読み込みと準備
# -------------------------------------------------------------
# どこから実行しても確実にファイルを読み込めるように、実行ファイルの位置を基準にします
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '../data/data.csv')
df = pd.read_csv(file_path)

# 目的変数（予測したいもの）の変換  
# 機械学習モデルは文字を理解できないため、'Yes'を1、'No'を0の「数値」に変換する
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# -------------------------------------------------------------
# 【設定】使う特徴量と、その影響力（重み）の相場を一括設定します
# -------------------------------------------------------------
# 「使う特徴量」と「その重み」をここでまとめて管理します。
# 1.0を基準とし、使わない特徴量はここの行ごと削除（またはコメントアウト）すればOKです。
FEATURE_WEIGHTS = {
    'OverTime': 2.5,          # [相場: 非常に高い 2.0〜3.0] 残業は肉体心理的負担への影響が最大
    'MonthlyIncome': 1.5,     # [相場: 高い 1.5〜2.0] 直接的な不満に直結しやすい
    'Age': 1.5,               # [相場: 高い 1.2〜1.8] 若手層の早期転職などが顕著なため
    'JobSatisfaction': 1.5,   # [相場: 高い 1.2〜1.5] アンケートなどのダイレクトな不満度
    'StressRating': 1.5,      # [相場: 高い 1.2〜1.5] 退職動機にそのまま繋がりやすい
    'YearsAtCompany': 1.2,    # [相場: 中程度 1.0〜1.2] ミスマッチからの早期離職を見る
    'TotalWorkingYears': 1.0, # [相場: 標準 1.0] キャリアの長さ
    'JobRole': 1.0,           # [相場: 標準 1.0] 職種（営業職など一部だけ離職率が高いケースがある）
    'NumCompaniesWorked': 0.8,# [相場: 控えめ 0.5〜0.8] 本人の性質(ジョブホッパー)の参考程度
    'DistanceFromHome': 0.8   # [相場: 控えめ 0.5〜0.8] 影響はあるが決定打にはなりにくい
}

# 上の設定から、自動的に「使う特徴量のリスト」を作成します
features = list(FEATURE_WEIGHTS.keys())

# モデルへの入力データ(X)と、予測したい正解ラベル(y)に分割する
X = df[features].copy()
y = df['Attrition']

# 'OverTime'などの文字データ(Yes/No など)を、0または1のダミー変数に変換する
X = pd.get_dummies(X, drop_first=True)

# -------------------------------------------------------------
# 2. データの分割と「標準化」（ニューラルネットワーク特有の必須準備）
# -------------------------------------------------------------
# 未知のデータに対する性能を測るため、一部（20%）を「テスト用（カンニングペーパー）」として隠す
# random_state=42 で乱数シードを固定し、誰が何度やっても同じように分割されるようにする
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # 離職率の割合が、学習用とテスト用で崩れないようにする工夫
)

# 【重要】スケール（単位）を揃える「標準化」
# 月収の「200,000」と年齢の「30」では数字の大きさが違いすぎます。
# ニューラルネットワークはこの違いに弱いため、平均0・分散1になるように単位を揃えます。
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 学習データから基準値を作って変換
X_test_scaled = scaler.transform(X_test)        # 作った基準値でテストデータも変換

# -------------------------------------------------------------
# 【追加】仮説に基づく特徴量の「重み付け」の適用
# -------------------------------------------------------------
# プログラム冒頭で一括設定した FEATURE_WEIGHTS の重みを、標準化済みのデータに掛け合わせます
columns_list = list(X.columns)
for feature_name, weight in FEATURE_WEIGHTS.items():
    # 職種(JobRole_Sales)のようにダミー化した名前にも重みを適用できるように判定します
    for col_idx, col_name in enumerate(columns_list):
        if col_name == feature_name or col_name.startswith(feature_name + '_'):
            X_train_scaled[:, col_idx] *= weight
            X_test_scaled[:, col_idx] *= weight

# -------------------------------------------------------------
# 3. モデルの作成と学習
# -------------------------------------------------------------
# 学習率とエポック数を簡単に変更できるように変数として用意します
LEARNING_RATE = 0.001  # 学習率（1回の学習でどのくらい賢くなるかの歩幅）
EPOCHS = 300           # エポック数（データ全体を何回繰り返し学習するか）

# WandB（Weights & Biases）の初期化と、今回試す設定の保存
# プロジェクト内で「どんな条件で学習したか」が一目で分かるように記録します
wandb.init(
    project="employee-attrition-prediction",  # WandB上のプロジェクト名
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "hidden_layers": (32, 16),
        "features_used": features
    }
)

# シンプルなニューラルネットワークモデル（MLPClassifier）を作る
# hidden_layer_sizes : 今回特徴量が増えたため、(32, 16) に拡大してモデルの表現力を上げます
# learning_rate_init : 用意した学習率の変数を設定
# max_iter : 用意したエポック数の変数を設定
# verbose=True : 学習中のエポック数や損失の推移を画面に出力して確認できるようにする
model = MLPClassifier(
    hidden_layer_sizes=(32, 16), 
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS, 
    random_state=42,
    verbose=True
)

# モデルに学習データを与えて、パターンを訓練（学習）させる
model.fit(X_train_scaled, y_train)

# -------------------------------------------------------------
# 4. モデルのテスト（評価）と、テストデータの正解率を出力
# -------------------------------------------------------------
# 学習したモデルに、隠しておいたテストデータを渡して予測させる
y_pred = model.predict(X_test_scaled)

# モデルの予測（y_pred）と実際の正解（y_test）を比較し、正解率を計算する
accuracy = accuracy_score(y_test, y_pred)
print(f"【テストデータの正解率】: {accuracy:.2%}")

# -------------------------------------------------------------
# 5. 学習の進み具合（エポック数と損失の関係）をグラフ化する
# -------------------------------------------------------------
# 損失（Loss）は「モデルの予測の誤差」のこと。
# エポック（学習回数）が進むにつれて誤差が右肩下がりに減っていくのが、良い学習ができている証拠です。
plt.figure(figsize=(8, 5))

# 学習によって記録された毎エポックの損失（loss_curve_）を折れ線グラフにプロット
plt.plot(model.loss_curve_, label='Training Loss (学習時の誤差)', color='blue')

# グラフの見た目を整える
plt.title('Loss Curve over Epochs (エポック数と損失の推移)')
plt.xlabel('Epochs (学習の反復回数)')
plt.ylabel('Loss (損失・誤差の大きさ)')
plt.legend()
plt.grid(True) # 目盛り線を表示

# グラフを表示
plt.show()

# -------------------------------------------------------------
# 6. Weights & Biases (WandB) への学習結果の書き込み
# -------------------------------------------------------------
# 学習後に保存された過去のエポック毎の誤差（loss_curve_）を、1エポックずつ記録します
for epoch, loss in enumerate(model.loss_curve_):
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": loss
    })

# テストデータで計算した最終的な正解率（Accuracy）も記録する
wandb.log({"final_accuracy": accuracy})

# -------------------------------------------------------------
# 7. 「どのように間違えたか（混同行列）」をWandBに記録
# -------------------------------------------------------------
# 本当は離職しない(0)のに離職する(1)と予測した数や、その逆など、間違いの具体的な傾向を可視化します
class_names = ["Not Attrition (No)", "Attrition (Yes)"]
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        preds=y_pred,
        y_true=y_test.values,
        class_names=class_names
    )
})

# WandBへの書き込みを終了する
wandb.finish()
