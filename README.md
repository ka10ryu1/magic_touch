# magic_touch

PINTO model zoo の[MagicTouch](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/391_MagicTouch)に矩形を追加したもの

## インストールから実行まで

```bash
$ git clone https://github.com/ka10ryu1/magic_touch.git
$ ./download.sh # AIモデルのダウンロード
$ python3 -m venv ENV # venvの環境設定
$ source ENV/bin/activate # venv有効化
$ pip install opencv-python onnxruntime # 関連ライブラリのインストール
$ mkdir data # dataフォルダを作って適当に画像を置いてください
$ ./demo_MagicTouch_onnx.py --image data/sample.jpg --model magic_touch.onnx # 素のデモ（[Esc]で終了）
$ ./demo_MagicTouch_onnx_v2.py --image data/sample.jpg --model magic_touch.onnx # 矩形を追加したデモ（[Esc]で終了）
```

## 解説

### ONNX について

- ONNX は AI モデルをフレームワークに依存せずに実行できるライブラリ
- 様々なフレームワークで AI モデルの ONNX エクスポート機能を搭載しているが、完全ではないことも多い（PINTO さんががんばって変換スクリプトを作っている）

以下のようにしてセッションを読み込めば、CPU でも GPU でもよしなに動いてくれる。

```python
# Load model
onnx_session = onnxruntime.InferenceSession(
    model_path,
    providers=[
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ],
)
```

推論部分はこれだけ。めちゃ簡単。

```python
# Inference
input_name = onnx_session.get_inputs()[0].name
result = onnx_session.run(None, {input_name: input_image})
```

### 矩形を追加（今回新規で書いたのはこれだけ）

```python
mask_size = np.sum(mask) / 255
# 小さい物体は誤検出として無視する
if mask_size > 500:
    continue

# 矩形抽出
contours, _ = cv.findContours(
    cv.cvtColor(mask, cv.COLOR_RGB2GRAY),
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_NONE
)
# 矩形の面積を計算して最大の矩形を抽出
x, y, w, h = cv.boundingRect(
    max(contours, key=lambda x: cv.contourArea(x))
)
# 矩形描画
cv.rectangle(debug_image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 5)
```
