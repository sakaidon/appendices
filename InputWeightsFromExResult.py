# Python3
# 実験結果を基にしたリザバーコンピューティングパラメータ（入力結合重み行列）の設定

import numpy as np

# 実験結果を基に作成した時間発展行列を受け取り，入力結合重み行列を返す関数
# 引数：ndarray，返り値：ndarray
def get_input_weights(arr):
    max = 0
    max_index = 0
    for i in range(len(arr)):
        if max < sum(arr[i]):
            max = sum(arr[i])
            max_index = i
    return arr[max_index] / np.max(arr[max_index]) # 最大値が1になるように正規化してreturn