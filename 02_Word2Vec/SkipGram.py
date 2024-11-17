# 第1步 构建实验语料库
# 定义一个句子列表，后面会用这些句子来训练 CBOW 和 Skip-Gram 模型
sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
             "Xiaobing is Student", "Xiaoxue is Student", ]
# 将所有句子连接在一起，然后用空格分隔成多个单词
words = ' '.join(sentences).split()
# 构建词汇表，去除重复的词
word_list = list(set(words))
# 创建一个字典，将每个词映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建一个字典，将每个索引映射到对应的词
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)  # 计算词汇表的大小
print(" 词汇表：", word_list)  # 输出词汇表
print(" 词汇到索引的字典：", word_to_idx)  # 输出词汇到索引的字典
print(" 索引到词汇的字典：", idx_to_word)  # 输出索引到词汇的字典
print(" 词汇表大小：", voc_size)  # 输出词汇表大小
'''
词汇表： ['Kage', 'Niuzong', 'Xiaoxue', 'Mazong', 'Xiaobing', 'Teacher', 'is', 'Boss', 'Student']
词汇到索引的字典： {'Kage': 0, 'Niuzong': 1, 'Xiaoxue': 2, 'Mazong': 3, 'Xiaobing': 4, 'Teacher': 5, 'is': 6, 'Boss': 7, 'Student': 8}
索引到词汇的字典： {0: 'Kage', 1: 'Niuzong', 2: 'Xiaoxue', 3: 'Mazong', 4: 'Xiaobing', 5: 'Teacher', 6: 'is', 7: 'Boss', 8: 'Student'}
词汇表大小： 9
'''


# 第2步 生成 Skip-Gram 训练数据
def create_skipgram_dataset(sentences, window_size=2):
    data = []  # 初始化数据
    for sentence in sentences:  # 遍历句子
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取相邻的单词，将当前单词前后各 N 个单词作为相邻单词
            for neighbor in sentence[max(idx - window_size, 0): min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:  # 排除当前单词本身
                    # 将相邻单词与当前单词作为一组训练数据
                    # data.append((neighbor, word))
                    data.append((word, neighbor))
    return data


# 使用函数创建 Skip-Gram 训练数据
skipgram_data = create_skipgram_dataset(sentences)
# 打印未编码的 Skip-Gram 数据样例（前 3 个）
print("Skip-Gram 数据样例（未编码）：", skipgram_data[:3])
print(skipgram_data)
'''
Skip-Gram 数据样例（未编码）： [('is', 'Kage'), ('Teacher', 'Kage'), ('Kage', 'is')]
'''

# 第3步 定义 One-Hot 编码函数
import torch  # 导入 torch 库


def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx))  # 创建一个长度与词汇表相同的全 0 张量
    tensor[word_to_idx[word]] = 1  # 将对应词的索引设为 1
    return tensor  # 返回生成的 One-Hot 向量


# 展示 One-Hot 编码前后的数据
word_example = "Teacher"
print("One-Hot 编码前的单词：", word_example)
print("One-Hot 编码后的向量：", one_hot_encoding(word_example, word_to_idx))
# 展示编码后的 Skip-Gram 训练数据样例
print("Skip-Gram数据样例（已编码）：", [(one_hot_encoding(target, word_to_idx),
                                      word_to_idx[context]) for context, target in skipgram_data[:3]])
'''
One-Hot 编码前的单词： Teacher
One-Hot 编码后的向量： tensor([0., 0., 0., 0., 0., 1., 0., 0., 0.])
Skip-Gram数据样例（已编码）： [(tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]), 6), (tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]), 5), (tensor([0., 0., 0., 0., 0., 0., 1., 0., 0.]), 0)]
'''

# 第4步 定义 Skip-Gram 类
import torch.nn as nn  # 导入 neural network


class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        # 从词汇表大小到嵌入层大小（维度）的线性层（权重矩阵）
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        # 从嵌入层大小（维度）到词汇表大小的线性层（权重矩阵）
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):  # 前向传播的方式，X 形状为 (batch_size, voc_size)
        # 通过隐藏层，hidden 形状为 (batch_size, embedding_size)
        hidden = self.input_to_hidden(X)
        # 通过输出层，output_layer 形状为 (batch_size, voc_size)
        output = self.hidden_to_output(hidden)
        return output


embedding_size = 2  # 设定嵌入层的大小，这里选择 2 是为了方便展示
skipgram_model = SkipGram(voc_size, embedding_size)  # 实例化 Skip-Gram 模型
print("Skip-Gram 模型：", skipgram_model)
'''
Skip-Gram 模型： SkipGram(
  (input_to_hidden): Linear(in_features=9, out_features=2, bias=False)
  (hidden_to_output): Linear(in_features=2, out_features=9, bias=False)
)
'''

# 第5步 训练 Skip-Gram 类
learning_rate = 0.001  # 设置学习速率
epochs = 1000  # 设置训练轮次
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
import torch.optim as optim  # 导入随机梯度下降优化器

optimizer = optim.SGD(skipgram_model.parameters(), lr=learning_rate)
# 开始训练循环
loss_values = []  # 用于存储每轮的平均损失值
for epoch in range(epochs):
    loss_sum = 0  # 初始化损失值
    # for context, target in skipgram_data:
    for center_word, context in skipgram_data:
        X = one_hot_encoding(center_word, word_to_idx).float().unsqueeze(0)  # 将中心词转换为 One-Hot 向量
        y_true = torch.tensor([word_to_idx[context]], dtype=torch.long)  # 将周围词转换为索引值
        y_pred = skipgram_model(X)  # 计算预测值
        loss = criterion(y_pred, y_true)  # 计算损失
        loss_sum += loss.item()  # 累积损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    if (epoch + 1) % 100 == 0:  # 输出每 100 轮的损失，并记录损失
        print(f"Epoch: {epoch + 1}, Loss: {loss_sum / len(skipgram_data)}")
        loss_values.append(loss_sum / len(skipgram_data))
# 绘制训练损失曲线
import matplotlib.pyplot as plt  # 导入 matplotlib

# 绘制二维词向量图
plt.rcParams["font.family"] = ['SimHei']  # 用来设定字体样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(range(1, epochs // 100 + 1), loss_values)  # 绘图
plt.title(' 训练损失曲线 ')  # 图题
plt.xlabel(' 轮次 ')  # X 轴 Label
plt.ylabel(' 损失 ')  # Y 轴 Label
plt.show()  # 显示图

'''
Epoch: 100, Loss: 2.206050093968709
Epoch: 200, Loss: 2.1944000005722044
Epoch: 300, Loss: 2.182283655802409
Epoch: 400, Loss: 2.167587486902873
Epoch: 500, Loss: 2.147610100110372
Epoch: 600, Loss: 2.1189597725868223
Epoch: 700, Loss: 2.0784456014633177
Epoch: 800, Loss: 2.02604193687439
Epoch: 900, Loss: 1.9687333305676777
Epoch: 1000, Loss: 1.9178066511948904
'''

# 第6步 展示词向量
# 输出 Skip-Gram 习得的词嵌入
print("Skip-Gram 词嵌入：")
for word, idx in word_to_idx.items():  # 输出每个词的嵌入向量
    print(f"{word}: {skipgram_model.input_to_hidden.weight[:, idx].detach().numpy()}")

'''
Skip-Gram 词嵌入：
Kage: [0.61016065 0.7306773 ]
Niuzong: [-0.12686515  0.38882658]
Xiaoxue: [0.16116168 0.41416478]
Mazong: [-0.10384143  0.38075024]
Xiaobing: [0.30230978 0.45506486]
Teacher: [ 0.47190428 -0.12305872]
is: [-0.08106854 -0.480878  ]
Boss: [-0.23589657  1.1961255 ]
Student: [0.1537026  0.69835067]
'''

fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    # 获取每个单词的嵌入向量
    vec = skipgram_model.input_to_hidden.weight[:, idx].detach().numpy()
    ax.scatter(vec[0], vec[1])  # 在图中绘制嵌入向量的点
    ax.annotate(word, (vec[0], vec[1]), fontsize=12)  # 点旁添加单词标签
plt.title(' 二维词嵌入 ')  # 图题
plt.xlabel(' 向量维度 1')  # X 轴 Label
plt.ylabel(' 向量维度 2')  # Y 轴 Label
plt.show()  # 显示图
