
- 第2课 词的向量表示Word2Vec和Embedding
  - 2.1 词向量约等于词嵌入
  - 2.2 Word2Vec：CBOW模型和Skip-Gram模型
  - 2.3 CBOW模型的代码实现
  - 2.4 通过nn.Embedding来实现词嵌入


# 第2课 词的向量表示Word2Vec和Embedding
## 2.1 词向量约等于词嵌入
## 2.2 Word2Vec：CBOW模型和Skip-Gram模型
CBOW(Continuous Bag of Words, 连续词袋模型)：给定上下文来预测目标词。
Skip-Gram：通过给定目标词来预测上下文。

预测具体的词不是Word2Vec的目标，它的真正目标是通过调节神经网络参数学习词嵌入，以捕捉词汇表中词语之间的语义和语法关系，为下游的NLP任务提供丰富的表示。

## 2.3 CBOW模型的代码实现
## 2.4 通过nn.Embedding来实现词嵌入

