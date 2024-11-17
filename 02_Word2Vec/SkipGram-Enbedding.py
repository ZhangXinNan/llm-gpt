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
词汇表： ['Xiaoxue', 'Kage', 'Boss', 'Xiaobing', 'is', 'Niuzong', 'Teacher', 'Mazong', 'Student']
词汇到索引的字典： {'Xiaoxue': 0, 'Kage': 1, 'Boss': 2, 'Xiaobing': 3, 'is': 4, 'Niuzong': 5, 'Teacher': 6, 'Mazong': 7, 'Student': 8}
索引到词汇的字典： {0: 'Xiaoxue', 1: 'Kage', 2: 'Boss', 3: 'Xiaobing', 4: 'is', 5: 'Niuzong', 6: 'Teacher', 7: 'Mazong', 8: 'Student'}
词汇表大小： 9
'''


# 生成 Skip-Gram 训练数据
def create_skipgram_dataset(sentences, window_size=2):
    data = []  # 初始化数据
    for sentence in sentences:  # 遍历句子
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取相邻的单词，将当前单词前后各 N 个单词作为相邻单词
            for neighbor in sentence[max(idx - window_size, 0):
            min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:  # 排除当前单词本身
                    # 将相邻单词与当前单词作为一组训练数据
                    data.append((neighbor, word))
    return data


# 使用函数创建 Skip-Gram 训练数据
skipgram_data = create_skipgram_dataset(sentences)
# 打印未编码的 Skip-Gram 数据样例（前 3 个）
print("Skip-Gram 数据样例（未编码）：", skipgram_data[:3])
'''
Skip-Gram 数据样例（未编码）： [('is', 'Kage'), ('Teacher', 'Kage'), ('Kage', 'is')]
'''

# 定义 One-Hot 编码函数
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
print("Skip-Gram 数据样例（已编码）：", [(one_hot_encoding(context, word_to_idx),
                                       word_to_idx[target]) for context, target in skipgram_data[:3]])

'''
One-Hot 编码前的单词： Teacher
One-Hot 编码后的向量： tensor([0., 0., 0., 0., 0., 0., 1., 0., 0.])
Skip-Gram 数据样例（已编码）： [(tensor([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 1), (tensor([0., 0., 0., 0., 0., 0., 1., 0., 0.]), 1), (tensor([0., 1., 0., 0., 0., 0., 0., 0., 0.]), 4)]
'''

# 定义 Skip-Gram 模型
import torch.nn as nn  # 导入 neural network


class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        # 从词汇表大小到嵌入大小的嵌入层（权重矩阵）
        self.input_to_hidden = nn.Embedding(voc_size, embedding_size)
        # 从嵌入大小到词汇表大小的线性层（权重矩阵）
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        hidden_layer = self.input_to_hidden(X)  # 生成隐藏层：[batch_size, embedding_size]
        output_layer = self.hidden_to_output(hidden_layer)  # 生成输出层：[batch_size, voc_size]
        return output_layer


embedding_size = 2  # 设定嵌入层的大小，这里选择 2 是为了方便展示
skipgram_model = SkipGram(voc_size, embedding_size)  # 实例化 Skip-Gram 模型
print("Skip-Gram 模型：", skipgram_model)
'''
Skip-Gram 模型： SkipGram(
  (input_to_hidden): Embedding(9, 2)
  (hidden_to_output): Linear(in_features=2, out_features=9, bias=False)
)
'''

# 训练 Skip-Gram 类
learning_rate = 0.001  # 设置学习速率
epochs = 1000  # 设置训练轮次
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
import torch.optim as optim  # 导入随机梯度下降优化器

optimizer = optim.SGD(skipgram_model.parameters(), lr=learning_rate)
# 开始训练循环
loss_values = []  # 用于存储每轮的平均损失值
for epoch in range(epochs):
    loss_sum = 0  # 初始化损失值
    for context, target in skipgram_data:
        X = torch.tensor([word_to_idx[target]], dtype=torch.long)  # # 输入是中心词
        y_true = torch.tensor([word_to_idx[context]], dtype=torch.long)  # 目标词是周围词
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

# 输出 Skip-Gram 习得的词嵌入
print("Skip-Gram 词嵌入：")
for word, idx in word_to_idx.items():  # 输出每个词的嵌入向量
    print(f"{word}: {skipgram_model.input_to_hidden.weight[idx].detach().numpy()}")

'''
Skip-Gram 词嵌入：
Xiaoxue: [-1.185277  1.298675]
Kage: [ 0.6576963 -0.7966969]
Boss: [-0.45424542 -2.8292804 ]
Xiaobing: [0.15641613 0.47727743]
is: [-0.06019835  0.9512902 ]
Niuzong: [0.12958734 0.22813088]
Teacher: [-0.9217507 -1.2967763]
Mazong: [-0.41714072  0.65298873]
Student: [ 1.1987574 -1.3199298]
'''
# 绘制二维词向量图
plt.rcParams["font.family"] = ['SimHei']  # 用来设定字体样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    # 获取每个单词的嵌入向量
    vec = skipgram_model.input_to_hidden.weight[idx].detach().numpy()
    ax.scatter(vec[0], vec[1])  # 在图中绘制嵌入向量的点
    ax.annotate(word, (vec[0], vec[1]), fontsize=12)  # 点旁添加单词标签
plt.title(' 二维词嵌入 ')  # 图题
plt.xlabel(' 向量维度 1')  # X 轴 Label
plt.ylabel(' 向量维度 2')  # Y 轴 Label
plt.show()  # 显示图
