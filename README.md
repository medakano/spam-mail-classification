# 【練習問題】スパムメール分類/spam-mail-classification

https://signate.jp/competitions/104

![LB_screenshot](/resources/LeaderBoard_on_2022_05_24.png)
※2022/05/24 現在の記録

---
## 概要
 - SIGNATEで提供されている文書分類の練習問題に取り組んだ
 - huggingface transformers を使用
 - RoBERTa, DeBERTa, XLNet の3モデルを使用してアンサンブルした

## 環境
 - WSL2 + Docker + GPU
 - GPU: NVIDIA GeForce RTX 3060
 - 使用イメージ：pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

---
## 事前準備
data/original に、コンペで配布されているデータを配置する。zipファイルは展開して配置する。<br>
メモ：コンテナ作成コマンド<br>
```
docker run --name hogehoge -v [ホストマシンのマウント先]:[コンテナの絶対パス] --gpus all -it pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime /bin/bash
```
Windows環境の場合、絶対パスを指定しようとすると、D:\～ のようになるが、「:」が使えなかったので、マウント先の階層まで移動して対応した。<br>
--gpus のオプションを忘れるとコンテナからGPUが見えないので注意(1敗)

---
## 学習・推論の流れ
### 1. データセット作成
k-fold Cross-Validation を実施。<br>
trainデータをk個のサブセットに分割する。<br>
うち1つを検証用データに使用し、残りを訓練用データに使用する。<br>
各サブセットを1回ずつ検証用データとして使用することで、kパターンの異なるサブセットの組み合わせでモデルの訓練ができる。<br>
今回、k=5とし、各モデルをアンサンブルして最終的な推論を作成した。<br><br>
実行コマンド

```
python make_dataset.py
```
外部データの使用も試したが、今回はスコアの向上に繋がらなかった。

### 2. モデルの訓練
実行コマンド
```
. train.sh
```
huggingfaceの文書分類スクリプト、run_glue.pyを使用している。<br>
https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification<br><br>
使用モデルやハイパーパラメータはベタ書きしているので適宜修正する。<br>
0.5エポックごとにevaluationとチェックポイントの保存をし、最終的にベストモデルおよびラストモデルを保存している。<br>

### 3. 推論の実施
実行コマンド
```
. predict.sh
```
各ベストモデルを使用して推論を作成する。<br>
run_glue.pyに微修正を加え、分類結果でなくlogitsを出力するようにしている。<br>

### 4. アンサンブルの実行
実行コマンド
```
python merge_prediction.py
```
logitsの平均でアンサンブルして最終的な推論を作成。