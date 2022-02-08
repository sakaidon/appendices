# Python3
# 実験結果を基にしたリザバーコンピューティングパラメータ（入力結合重み行列）の設定

import numpy as np
import math
from scipy.optimize import curve_fit

TIME_PER_STEP = 2 # 1ステップあたりの時間(ns)

# フィッティング用の関数（ピーク値n，崩壊定数ramdaの指数減衰）
# 引数：float，float，float，返り値：float
def fit_func(t, n, ramda):
    return n * math.e ** (-ramda * t)

# x軸とy軸のデータセットを受け取り，上記のfit_func関数でフェッティングし，フィッティングパラメータ（n,ramda）のリストを返す関数
# 引数：ndarray，ndarray，返り値：list
def exp_fit(arr_x, arr_y):
    param, cov = curve_fit(fit_func, arr_x, arr_y, p0=[1600,2.0])
    return param

# 実験結果を基に作成した時間発展行列を受け取り，それぞれのノードの漏れ率の行列を返す関数
# 引数：list，返り値：list
def get_leak(arr, time):
    leak_list = []
    arr = np.array(arr).T
    for i in range(len(arr)):
        param = exp_fit(time[4:], arr[i,4:]) # 励起されたタイミングからフィッティング
        leak_list.append(np.exp(param[1]) / TIME_PER_STEP) # 蛍光寿命/1ステップあたりの時間を漏れ率として返り値の行列に追加
    return leak_list