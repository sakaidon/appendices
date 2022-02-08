# Python3
# リザバーコンピューティングによる時系列XOR問題近似

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import random

LEAK_RATE=0.7 # 漏れ率
NUM_RESERVOIR_NODES = 40 # リザバー層のノード数
DATA_LENGTH = 100 # 学習するデータ長

# データ長を受け取り，ランダムに0か1の並んだ入力データを作成する関数
# 引数：int，返り値：list
def generate_bits(length):
    ret_arr = []
    for i in range(length):
        ret_arr.append(random.randint(0,1))
    return ret_arr

# ビット列を受け取り，時間相関XOR問題にかけた結果を返す関数
# 引数：list，返り値：list
def generate_XOR(input):
    ret_arr = []
    for i in range(len(input)):
        if i==0:
            ret_arr.append(input[i])
            continue
        if input[i-1]^input[i]:
            ret_arr.append(1)
        else:
            ret_arr.append(0)
    return(ret_arr)

# 二つのlistを受け取り，0.5を閾値としたBERを返す関数
# 引数：list，list，返り値：float
def ret_BER(a,b):
    length = len(a)
    num_false = 0
    for i in range(length):
        if i == 0: continue
        elif a[i] >= 0.5 and b[i] >= 0.5: continue
        elif a[i] < 0.5 and b[i] < 0.5: continue
        else: num_false += 1
    return num_false / length

# リザバー層の活性化関数
# 引数：float，返り値：float
def activator(x):
    return np.tanh(x)

# 入力層，リザバー層，出力層をセットにしたクラス
class ReservoirNetWork:

    def __init__(self, inputs, teacher, test, num_input_nodes=1, num_reservoir_nodes=40, num_output_nodes=1, leak_rate=0.1, activator=activator):
        self.inputs = inputs
        self.teacher = teacher
        self.test = test
        self.log_reservoir_nodes = np.array([np.zeros(num_reservoir_nodes)])

        # init weights
        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])

        # それぞれの層のノードの数
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.num_output_nodes = num_output_nodes

        self.leak_rate = leak_rate
        self.activator = activator

    # reservoir層のノードの次の状態を取得
    def _get_next_reservoir_nodes(self, input, current_state):
        next_state = (1 - self.leak_rate) * current_state
        next_state += self.leak_rate * (np.array([input]) @ self.weights_input
            + current_state @ self.weights_reservoir)
        return self.activator(next_state)

    # 出力層の重みを更新
    def _update_weights_output(self, lambda0):
        # Ridge Regression
        E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0
        inv_x = np.linalg.inv(self.log_reservoir_nodes[1:].T @ self.log_reservoir_nodes[1:] + E_lambda0)
        # update weights of output layer
        self.weights_output = (inv_x @ self.log_reservoir_nodes[1:].T) @ self.teacher[1:]

    # 学習する
    def train(self, lambda0=0.0):
        for input in self.inputs:
            current_state = np.array(self.log_reservoir_nodes[-1])
            self.log_reservoir_nodes = np.append(self.log_reservoir_nodes,
                [self._get_next_reservoir_nodes(input, current_state)], axis=0)
        self.log_reservoir_nodes = self.log_reservoir_nodes[1:]
        self._update_weights_output(lambda0)

    # 学習で得られた重みを基に訓練データを学習できているかを出力
    def get_train_result(self):
        outputs = []
        reservoir_nodes = np.zeros(self.num_reservoir_nodes)
        for input in self.inputs:
            reservoir_nodes = self._get_next_reservoir_nodes(input, reservoir_nodes)
            outputs.append(self.get_output(reservoir_nodes))
        return outputs

    # 予測する
    def predict(self, test, lambda0=0.0):
        reservoir_nodes = np.zeros(self.num_reservoir_nodes)
        ret = []
        for i in range(len(test)):
            reservoir_nodes = self._get_next_reservoir_nodes(test[i], reservoir_nodes)
            ret.append(self.get_output(reservoir_nodes))
        return ret # 最初に使用した学習データの最後のデータを外して返す

    # get output of current state
    def get_output(self, reservoir_nodes):
        # return self.activator(reservoir_nodes @ self.weights_output)
        return reservoir_nodes @ self.weights_output

    # 重みを0.1か-0.1で初期化したものを返す
    def _generate_variational_weights(self, num_pre_nodes, num_post_nodes):
        # return (np.random.randint(0, 2, num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]) * 2 - 1) * 0.1
        return (np.random.random(num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]))

    # Reservoir層の重みを初期化
    def _generate_reservoir_weights(self, num_nodes):
        weights = np.random.random(num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        spectral_radius = max(abs(linalg.eigvals(weights)))
        return weights / spectral_radius

def main():
    input = generate_bits(DATA_LENGTH)
    test = generate_bits(DATA_LENGTH)
    train_teacher = generate_XOR(input)
    test_teacher = generate_XOR(test)

    model = ReservoirNetWork(inputs=input,
        teacher=train_teacher,
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        leak_rate=LEAK_RATE,
        test=test)

    model.train()
    train_result = model.get_train_result()
    test_result = model.predict(test)
    ber_train = ret_BER(train_teacher,train_result)
    ber_test = ret_BER(test_teacher,test_result)
    print(f"train_BER : {ber_train}\ntest_BER : {ber_test}")

    t = np.arange(1, DATA_LENGTH+1) # プロット用横軸

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(t, test_teacher, label="Teacher", color="red")
    ax1.plot(t, test_result, label="Output", color="blue")
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1).get_frame().set_alpha(1.0)
    ax1.set_ylim(-0.1,1.1)
    plt.show()

if __name__=="__main__":
    main()