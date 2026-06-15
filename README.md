# DeepLabV3 セマンティックセグメンテーション (MNIST 合成データ版)

DeepLabV3-ResNet50 によるピクセル単位 multi-label セマンティックセグメンテーションのデモ実装である。
研究室で取り組んでいる **プラズマ波動(ヒス・コーラス)のスペクトログラム上での自動検出パイプライン** から、
データポリシー上公開できない部分を全て差し替え、技術的本質のみを残した版である。

## プロジェクトの位置づけ

筆者は研究室で、JAXA/ISAS および 名古屋大学 ISEE が運用する あらせ衛星 (ERG) の
PWE/OFA-SPEC 磁界スペクトログラムを対象に、プラズマ波動 (ヒス・コーラス) を
セマンティックセグメンテーションで検出するパイプラインを開発した。
このリポジトリは、研究室成果物のうち以下を **全て差し替え** た公開版である:

- **学習データ**: ERG/Arase 実データ → MNIST 合成画像
- **正解ラベル**: 研究室先輩によるアノテーション → MNIST ラベルから自動生成した multi-hot マスク
- **学習済み重み**: 公開しない (実データの統計を間接的に保持する可能性があるため)

これは ERG プロジェクトの [Rules of the Road](https://ergsc.isee.nagoya-u.ac.jp/data_info/erg.shtml.ja)
で実データおよびその派生物の再配布が禁止されているためであり、また先輩のアノテーション知見を
無断公開すべきでないという研究倫理上の判断による。

## なぜ MNIST で本物のタスクと同等のスキルが示せるのか

プラズマ波動セグメンテーションと MNIST 合成セグメンテーションは、以下の本質的構造を共有している:

| 構造 | プラズマ波動 (実問題) | MNIST 合成 (本リポジトリ) |
|---|---|---|
| 入力形状 | 縦長スペクトログラム (時間×周波数) | 縦長キャンバス 1024×180 |
| 前景の疎性 | 強度の高いパッチが疎に存在 | 数字が疎に配置 |
| ノイズ背景 | 機器熱雑音、機器の再起動・振動雑音、その他機器ノイズ | 1/f ノイズ、縦ストリーク、横バンド、ベジエ曲線 |
| クラス間重なり | ヒスとコーラスが周波数帯で重なりうる | 数字同士の重なり許可、multi-hot ラベル |
| クラス数 | 背景 + 2 種類の波動 | 背景 + 10 種類の数字 |
| 評価指標 | クラスごとの F1 | クラスごとの F1 (同じ) |

ノイズ 3 種類(1/f、縦ストリーク、横バンド)は実スペクトログラムで観察される
3 類型のノイズ(機器熱雑音、機器の再起動・振動雑音、定常的な機器ノイズ)に対応するよう
意図的に設計している。詳細は `dataset_generator.py` のコメントを参照。

## ベース実装

本ノートブックは、研究室で開発した実データ版 `DeepLabV3_ver7_データ拡張.ipynb` をベースにしている。
変更点は冒頭のマークダウンセルに表形式で記載している。

## ファイル構成

```
.
├── DeepLabV3_MNIST_segmentation.ipynb   # メインノートブック
├── dataset_generator.py                  # 合成データセット生成モジュール
└── README.md                             # 本ファイル
```

実行時に以下が自動生成される:
```
├── mnist_data/                           # MNIST 原本 (torchvision がダウンロード)
├── data_mnist_synthetic/
│   ├── train_val_pool/                   # 訓練+検証用合成データ (8:2 分割)
│   │   ├── images/000000.npy             # (1024, 180) uint8
│   │   └── masks/000000.npy              # (11, 1024, 180) uint8
│   └── test/                             # 汎化評価用 (MNIST テスト分から生成)
└── saved_model/
    └── DeepLabV3_MNIST_segmentation_outputs/
        ├── YYYYMMDD_best_model.pth
        ├── YYYYMMDD_deeplabv3_mnist.onnx
        └── YYYYMMDD_output_visualization/
```

## 実行手順

```bash
pip install torch torchvision pillow numpy matplotlib scikit-learn tqdm onnx onnxruntime ipynbname
jupyter notebook DeepLabV3_MNIST_segmentation.ipynb
```

> **動作確認環境**: `torch==1.13.1+cu117` 前提。

ノートブックを上から順に実行する。
GPU が無い環境では Cell 2 の `CUDA_VISIBLE_DEVICES` 指定を外し、`DEVICE = 'cpu'` で動作する
(学習速度は大幅に低下する)。

## モデル

- アーキテクチャ: DeepLabV3 (ResNet50 backbone)
- 入力: 1ch グレースケール (`backbone.conv1` を `Conv2d(1, 64, ...)` に差し替え)
- 出力: 11 チャネル logits (background + 数字 0〜9)
- 損失: `BCEWithLogitsLoss`(チャネル毎の `pos_weight` で不均衡補正)
- 推論判定: `sigmoid(logits) > 0.5`
- ONNX 出力: **logits のまま (sigmoid は含めない)**。後段で閾値を変えられるように意図的に分離している。

## 結果と考察



## 既知の制限・改善余地

- 学習エポック数は 10 で初期実装。本リポジトリの主目的はデモであり、ハイパーパラメータ最適化は行っていない。
- ベジエ曲線のパラメータ(本数 0〜3、制御点 3〜4 個)は経験則的に設定したものであり、
  これがモデルの「紛らわしさ耐性」をテストするのに最適かは確認していない。
- 数字スケールは MNIST元画像の3.0 倍固定で実装している。複数スケール対応(連続一様分布)はパラメータ
  `DEFAULT_SCALE_RANGE` の変更で実現可能だが、現実装では検証していない。

## 引用・参考

- DeepLabV3: Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
- ERG project: Miyoshi, Shinohara et al., Earth Planets Space, DOI:10.1186/s40623-018-0862-0, 2018
- ERG Science Center: https://ergsc.isee.nagoya-u.ac.jp/index.shtml.en
