# Python3
# 時間–空間蛍光出力　→ 時間発展行列

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

LONGTH_TIME_NS = 10 # 時間–空間蛍光出力の時間方向の長さ（ns）
LONGTH_SPACE_UM = 4000 # 時間–空間蛍光出力の空間方向の長さ（µm）
TIME_PER_NODE = 0.5 # 時間発展行列の1要素あたりの時間（ns）
SCALE_PER_NODE = 100 # 時間発展行列の1要素あたりの空間（µm）

raw_data_dir = 'raw_data_dir_path' # 時間–空間蛍光出力ファイルのあるディレクトリ
result_data_dir = 'result_data_dir_path' # 処理後の時間発展行列を保存するディレクトリ
num_time_pixels = 968 # 時間–空間蛍光出力の時間方向のピクセル数
num_space_pixels = 1280 # 時間–空間蛍光出力の空間方向のピクセル数
num_time_node = int(LONGTH_TIME_NS // TIME_PER_NODE) # 処理後の時間発展行列のステップ数
num_space_node = LONGTH_SPACE_UM // SCALE_PER_NODE # 処理後の時間発展行列のノード数
pixels_per_time_node = num_time_pixels // num_time_node # 一ステップに相当する時間–空間蛍光出力の時間方向のピクセル数
pixels_per_space_node = num_space_pixels // num_space_node # 一ノードに相当する時間–空間蛍光出力の空間方向のピクセル数

# 時間–空間蛍光出力ファイルを読み込んでndarray形式で返す関数
# 引数：txtファイルのパス，返り値：ndarray
def file2arr(file_name):
  ret_arr = []
  tmp_file = open(file_name, "r", encoding="utf_8")
  while True:
    line = tmp_file.readline()
    ret_arr.append([])
    if line:
      tmp_line = line.replace("\n","").split("\t")
      for i in range(len(tmp_line)):
        ret_arr[-1].append(float(tmp_line[i]))
    else:
      break
  if len(ret_arr[-1]) != len(ret_arr[-2]): del(ret_arr[-1])
  return(np.array(ret_arr))

# 上で定義した1ステップあたりの時間と1ノードあたりの空間に基づいて時間–空間蛍光出力を時間発展行列に処理する関数
# 引数：ndarray，返り値：ndarray
def arr2nodes(arr):
  ret_arr = np.zeros([num_time_node, num_space_node])
  for i in range(len(ret_arr)):
    for j in range(len(ret_arr[0])):
      for k in range(pixels_per_time_node):
        for l in range(pixels_per_space_node):
          ret_arr[i][j] += arr[pixels_per_time_node*i + k][pixels_per_space_node*j + l]
  return ret_arr

# 時間発展行列のヒートマップを作成し，上で定義した保存ディレクトリに指定したファイル名で保存する関数
# 引数：ndarray，str，返り値：なし
def arr2heatmap(arr, save_name):
  for i in range(len(arr)):
    for j in range(len(arr[0])):
      if arr[i][j] < 0: arr[i][j] = 0

  fig = plt.figure()
  ax1 = fig.add_subplot(1, 1, 1)
  sns.heatmap(arr, ax=ax1, cmap='gist_heat')
  ax1.set_xlabel('Position [mm]')
  ax1.set_ylabel('Time [nm]')
  ax1.set_xticks([0,len(arr[0])/2,len(arr[0])]) 
  ax1.set_yticks([0,len(arr)/10,2*len(arr)/10,3*len(arr)/10,4*len(arr)/10,5*len(arr)/10,6*len(arr)/10,7*len(arr)/10,8*len(arr)/10,9*len(arr)/10,len(arr)]) 
  ax1.set_xticklabels(['0.0','2.0','4.0'], rotation=0)
  ax1.set_yticklabels(['0','1','2','3','4','5','6','7','8','9','10'], rotation=0)
  ax2 = ax1.twiny()
  ax2.set_xlabel('Nodes')
  ax2.set_xticks([0,0.5,1.0]) 
  ax2.set_xticklabels(['0',f'{int(num_space_node/2)}',f'{num_space_node}'])

  plt.savefig(result_data_dir + save_name + ".png", bbox_inches='tight', pad_inches=0, format='png', dpi=300)

def main():
    raw_data_list = glob.glob(raw_data_dir + '*.txt') # raw_data_dir内のtxtファイルのパスのリスト

    # raw_data_listの全要素に対する処理
    for i in range(len(raw_data_list)):
        tmp_save_name = raw_data_list[i].replace(raw_data_dir,"").replace(".txt","")
        tmp_arr = file2arr(raw_data_list[i])
        tmp_nodes = arr2nodes(tmp_arr)
        arr2heatmap(tmp_nodes,tmp_save_name) # 時間発展行列をpng形式のヒートマップで保存
        df = pd.DataFrame(tmp_nodes)
        df.to_csv(result_data_dir + tmp_save_name + '.csv') # 時間発展行列をcsv形式で保存

if __name__=="__main__":
    main()