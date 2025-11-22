
import numpy as np
import matplotlib.pyplot as plt
#インターネット上のファイルをダウンロード
import urllib.request
#オペレーティングシステムと連携し、ファイルなどの操作
import os

class neuralNetwork:

    def __init__(
            self,
            input_neurons,  # 入力層のニューロン数
            hidden_neurons,  # 隠れ層のニューロン数
            output_neurons,  # 出力層のニューロン数
            learning_rate  # 学習率
    ):
        '''
        ニューラルネットワークの初期化を行う

        '''
        # 入力層、隠れ層、出力層のニューロン数をインスタンス変数に代入
        self.inneurons = input_neurons  # 入力層のニューロン数
        self.hneurons = hidden_neurons  # 隠れ層のニューロン数
        self.oneurons = output_neurons  # 出力層のニューロン数
        self.lr = learning_rate  # 学習率
        self.weight_initializer()  # weight_initializer()を呼ぶ

    def weight_initializer(self):
        '''
        重みとバイアスの初期化を行う

        '''
        # 隠れ層の重みとバイアスを初期化
        self.w1 = np.random.normal(
            0.0,  # 平均は0
            pow(self.inneurons, -0.5),  # 標準偏差は入力層のニューロン数を元に計算
            (self.hneurons,  # 行数は隠れ層のニューロン数
             self.inneurons + 1)  # 列数は入力層のニューロン数 + 1
        )

        # 出力層の重みとバイアスを初期化
        self.w2 = np.random.normal(
            0.0,  # 平均は0
            pow(self.hneurons, -0.5),  # 標準偏差は隠れ層のニューロン数を元に計算
            (self.oneurons,  # 行数は出力層のニューロン数
             self.hneurons + 1)  # 列数は隠れ層のニューロン数 + 1
        )

    def sigmoid(self, x):
        '''
        シグモイド関数
        ------------------------
        x : 関数を適用するデータ
        '''
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        '''
        ソフトマックス関数
        ------------------------
        x : 関数を適用するデータ
        '''
        c = np.max(x)
        exp_x = np.exp(x - c)  # オーバーフローを防止する
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y

    def train(self, inputs_list, targets_list):
        '''
        ニューラルネットワークの学習を行う
        ------------------------
        inputs_list  : 訓練データの配列
        targets_list : 正解ラベルの配列

        '''
        ## [入力層]
        # 入力値の配列にバイアス項を追加して入力層から出力する
        inputs = np.array(
            np.append(
                inputs_list, [1]),  # 配列の末尾にバイアスのための「1」を追加
            ndmin=2  # 2次元化
        ).T  # 転置して1列の行列にする

        ## [隠れ層]
        # 入力層の出力に重み、バイアスを適用して隠れ層に入力する
        hidden_inputs = np.dot(
            self.w1,  # 隠れ層の重み
            inputs  # 入力層の出力
        )
        # シグモイド関数を適用して隠れ層から出力
        hidden_outputs = self.sigmoid(hidden_inputs)
        # 隠れ層の出力行列の末尾にバイアスのための「1」を追加
        hidden_outputs = np.append(
            hidden_outputs,  # 隠れ層の出力行列
            [[1]],  # 2次元形式でバイアス値を追加
            axis=0  # 行を指定(列は1)
        )

        ## [出力層]
        # 出力層への入力信号を計算
        final_inputs = np.dot(
            self.w2,  # 隠れ層と出力層の間の重み
            hidden_outputs  # 隠れ層の出力
        )
        # ソフトマックス関数を適用して出力層から出力する
        final_outputs = self.softmax(final_inputs)

        ## ---バックプロパゲーション---(出力層)
        # 正解ラベルの配列を1列の行列に変換する
        targets = np.array(
            targets_list,  # 正解ラベルの配列
            ndmin=2  # 2次元化
        ).T  # 転置して1列の行列にする
        # 出力値と正解ラベルとの誤差
        output_errors = final_outputs - targets
        # 出力層の入力誤差δを求める
        delta_output = output_errors * (1 - final_outputs) * final_outputs
        # 重みを更新する前に隠れ層の出力誤差を求めておく
        hidden_errors = np.dot(
            self.w2.T,  # 出力層の重み行列を転置する
            delta_output  # 出力層の入力誤差δ
        )
        # 出力層の重み、バイアスの更新
        self.w2 -= self.lr * np.dot(
            # 出力誤差＊(1－出力信号)＊出力信号
            delta_output,
            # 隠れ層の出力行列を転置
            hidden_outputs.T
        )
        ## ---バックプロパゲーション---(隠れ層)
        # 逆伝搬された隠れ層の出力誤差からバイアスのものを取り除く
        hidden_errors_nobias = np.delete(
            hidden_errors,  # 隠れ層のエラーの行列
            self.hneurons,  # 隠れ層のニューロン数をインデックスにする
            axis=0  # 行の削除を指定
        )
        # 隠れ層の出力行列からバイアスを除く
        hidden_outputs_nobias = np.delete(
            hidden_outputs,  # 隠れ層の出力の行列
            self.hneurons,  # 隠れ層のニューロン数をインデックスにする
            axis=0  # 行の削除を指定
        )
        # 隠れ層の重み、バイアスの更新
        self.w1 -= self.lr * np.dot(
            # 逆伝搬された隠れ層の出力誤差＊(1－隠れ層の出力)＊隠れ層の出力
            hidden_errors_nobias * (1.0 - hidden_outputs_nobias
                                    ) * hidden_outputs_nobias,
            # 入力層の出力信号の行列を転置
            inputs.T
        )

    def evaluate(self,
                     inputs_list
                     ):
            '''
            学習した重みでテストデータを評価する
            ------------------------------------
            inputs_list : テスト用データの配列

            '''
            ## [入力層]
            # 入力値の配列にバイアス項を追加して入力層から出力する
            inputs = np.array(
                np.append(inputs_list, [1]),  # 配列の末尾にバイアスの値「1」を追加
                ndmin=2  # 2次元化
            ).T  # 転置して1列の行列にする

            ## [隠れ層]
            # 入力層の出力に重み、バイアスを適用して隠れ層に入力する
            hidden_inputs = np.dot(self.w1,  # 入力層と隠れ層の間の重み
                                   inputs  # テストデータの行列
                                   )
            # 活性化関数を適用して隠れ層から出力する
            hidden_outputs = self.sigmoid(hidden_inputs)
            ## [出力層]
            # 出力層への入力信号を計算
            final_inputs = np.dot(
                self.w2,  # 隠れ層と出力層の間の重み
                np.append(hidden_outputs, [1]),  # 隠れ層の出力配列の末尾にバイアスの値「1」を追加
            )
            # 活性化関数を適用して出力層から出力する
            final_outputs = self.softmax(final_inputs)

            return final_outputs




data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 'train-images.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
                           filename=os.path.join(data_folder, 'train-labels.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 'test-images.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz',
                           filename=os.path.join(data_folder, 'test-labels.gz'))
import gzip
import struct

# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = load_data(os.path.join(
    data_folder, 'train-images.gz'), False) / 255.
X_train = (X_train*0.99) + 0.01
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
X_test = (X_test*0.99) + 0.01
y_train = load_data(os.path.join(
    data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(
    data_folder, 'test-labels.gz'), True).reshape(-1)
#時間を測定するモジュールをインポート
import time
#プログラムの開始時刻を取得
start = time.time()

input_neurons = 784     #入力層のニューロンの数
hidden_neurons = 200    #隠れ層のニューロンの数
output_neurons = 10     #出力層のニューロンの数
learning_rate = 0.2     #学習率

#neuralNetworkクラスのインスタンス化
n = neuralNetwork(input_neurons,
                  hidden_neurons,
                  output_neurons,
                  learning_rate)

#ニューラルネットワークの学習
epocks = 5              #学習を繰り返す回数

#指定した回数だけ学習を繰り返す
for e in range(epocks):
    #画像データと正解ラベルを１セットずつ取り出す
    for(inputs,target) in zip(X_train,y_train):
        #出力層のニューロン数を要素数とするOne-hotベクトルを作成
        targets = np.zeros(output_neurons) + 0.01
        #正解ラベルに対応する要素を0.99にする
        targets[int(target)] = 0.99
        #画像データと正解ラベルの1セットを引数にしてtrain()を実行
        n.train(inputs,     #訓練データの行列
                targets     #目標値の行列
                )
print("Computation time:{0:3f} sec".format(time.time() - start))


#正解は1,不正解は0を格納する配列
score = []

#X_testをinputs,y_testをcorrect_labelに格納
for(inputs,correct_label) in zip(X_test,y_test):
    #ニューラルネットワークで評価する
    outputs = n.evaluate(inputs)
    #出力層のニューロン数に合わせて正解の配列を作成
    targets = np.zeros(output_neurons) + 0.01
    #正解値に対応する要素を0.99にする
    targets[int(correct_label)] = 0.99
    #出力の行列の最大値のインデックスが予測する手書き数字に対応
    label = np.argmax(outputs)
    #ネットワークの出力と正解ラベルを比較
    if(label == correct_label):
        score.append(1)     #正解ならscoreに1を追加
    else:
        score.append(0)     #不正解なら0を追加

result = ['○'if i == 1 else '●' for i in score]
print(result)

score_arry = np.asarray(score)

print("performance = ",score_arry.sum() / score_arry.size)

