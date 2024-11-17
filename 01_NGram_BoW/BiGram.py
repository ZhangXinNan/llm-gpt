# 构建一个玩具数据集
corpus = ["我喜欢吃苹果",
          "我喜欢吃香蕉",
          "她喜欢吃葡萄",
          "他不喜欢吃香蕉",
          "他喜欢吃苹果",
          "她喜欢吃草莓"]


# 定义一个分词函数，将文本转换为单个字符的列表
def tokenize(text):
    return [char for char in text]  # 将文本拆分为字符列表


# 对每个文本进行分词，并打印出对应的单字列表
print("单字列表:")
for text in corpus:
    tokens = tokenize(text)
    print(tokens)

# 定义计算 N-Gram 词频的函数
from collections import defaultdict, Counter  # 导入所需库


def count_ngrams(corpus, n):
    ngrams_count = defaultdict(Counter)  # 创建一个字典，存储 N-Gram 计数
    for text in corpus:  # 遍历语料库中的每个文本
        tokens = tokenize(text)  # 对文本进行分词
        for i in range(len(tokens) - n + 1):  # 遍历分词结果，生成 N-Gram
            ngram = tuple(tokens[i:i + n])  # 创建一个 N-Gram 元组
            prefix = ngram[:-1]  # 获取 N-Gram 的前缀
            token = ngram[-1]  # 获取 N-Gram 的目标单字
            ngrams_count[prefix][token] += 1  # 更新 N-Gram 计数
    return ngrams_count


bigram_counts = count_ngrams(corpus, 2)  # 计算 bigram 词频
print("bigram 词频：")  # 打印 bigram 词频
for prefix, counts in bigram_counts.items():
    print("{}: {}".format("".join(prefix), dict(counts)))

'''
bigram 词频：
我: {'喜': 2}
喜: {'欢': 6}
欢: {'吃': 6}
吃: {'苹': 2, '香': 2, '葡': 1, '草': 1}
苹: {'果': 2}
香: {'蕉': 2}
她: {'喜': 2}
葡: {'萄': 1}
他: {'不': 1, '喜': 1}
不: {'喜': 1}
草: {'莓': 1}
'''


# 定义计算 N-Gram 出现概率的函数
def ngram_probabilities(ngram_counts):
    ngram_probs = defaultdict(Counter)  # 创建一个字典，存储 N-Gram 出现的概率
    for prefix, tokens_count in ngram_counts.items():  # 遍历 N-Gram 前缀
        total_count = sum(tokens_count.values())  # 计算当前前缀的 N-Gram 计数
        for token, count in tokens_count.items():  # 遍历每个前缀的 N-Gram
            ngram_probs[prefix][token] = count / total_count  # 计算每个 N-Gram 出现的概率
    return ngram_probs


bigram_probs = ngram_probabilities(bigram_counts)  # 计算 bigram 出现的概率
print("\nbigram 出现的概率 :")  # 打印 bigram 概率
for prefix, probs in bigram_probs.items():
    print("{}: {}".format("".join(prefix), dict(probs)))

'''
bigram 出现的概率 :
我: {'喜': 1.0}
喜: {'欢': 1.0}
欢: {'吃': 1.0}
吃: {'苹': 0.3333333333333333, '香': 0.3333333333333333, '葡': 0.16666666666666666, '草': 0.16666666666666666}
苹: {'果': 1.0}
香: {'蕉': 1.0}
她: {'喜': 1.0}
葡: {'萄': 1.0}
他: {'不': 0.5, '喜': 0.5}
不: {'喜': 1.0}
草: {'莓': 1.0}
'''


# 定义生成下一个词的函数
def generate_next_token(prefix, ngram_probs):
    if not prefix in ngram_probs:  # 如果前缀不在 N-Gram 中，返回 None
        return None
    next_token_probs = ngram_probs[prefix]  # 获取当前前缀的下一个词的概率
    next_token = max(next_token_probs,
                     key=next_token_probs.get)  # 选择概率最大的词作为下一个词
    return next_token


# 定义生成连续文本的函数
def generate_text(prefix, ngram_probs, n, length=6):
    tokens = list(prefix)  # 将前缀转换为字符列表
    for _ in range(length - len(prefix)):  # 根据指定长度生成文本
        # 获取当前前缀的下一个词
        next_token = generate_next_token(tuple(tokens[-(n - 1):]), ngram_probs)
        if not next_token:  # 如果下一个词为 None，跳出循环
            break
        tokens.append(next_token)  # 将下一个词添加到生成的文本中
    return "".join(tokens)  # 将字符列表连接成字符串


# 输入一个前缀，生成文本
generated_text = generate_text("我", bigram_probs, 2)
print("\n 生成的文本：", generated_text)  # 打印生成的文本

# 生成的文本： 我喜欢吃苹果
