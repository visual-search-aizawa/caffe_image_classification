# Caffe用簡単画像分類スクリプト

---


これはCNNの中身を考えずになんとなくDeep Learningを使ってみたい人向けのスクリプトです．

## 必要なもの
事前に
```console
$ caffe.bin
```
でCNNが実行できる環境（wscript内の該当箇所を変更できれば実行方法はこれに限らない）

## 内容物
- maf.py
- waf
	- 以上ふたつは実験用のおまじない的スクリプト．詳細はここでは省く．mafについては[こちら](https://github.com/pfi/maf/)wafについては[こちら](https://github.com/waf-project/waf)を参照
- wscript
- train_val.prototxt
- data
	- 空ディレクトリ
	
## 自分のデータを分類する
1. data以下に，クラスごとに分けた画像たちを入れる
2. train_val.prototxtの最終層のnum_outputというパラメータを分類クラス数に合わせて変える
3. wscriptの以下の部分を変更する

| 箇所      |     説明 |  
| :--------:| :--------:| 
| ３つめのexp，train_ratio   | データの内学習に使うデータの割合（デフォ0.5) |  
|7つ目のexp, train_batchsize, test_batch_size|学習テストそれぞれのミニバッチサイズ|
|8つめのexpの各パラメータ|base_lrが学習率．詳しくはhttp://caffe.berkeleyvision.org/tutorial/solver.html|

オプションとしては，train_val.prototxtでネットワークの構成変更が可能です．また，fine-tuneしたいときはコードの中のfor finetuneと書いた部分を参考にしてください．

## 実行
```console
$ ./waf configure
$ ./waf
```
## その後
自動的にbuildフォルダが生成され，その中に中間ファイルなどが保存されます．caffeのlogはbuild/log以下に保存されます．
```console
$ tail -f build/log/1-log.txt
```
等で追えます

## わからなければ
M2加賀谷orM1堀口まで
