# 样例选择器
# 自定义、长度选样、MMR样例选择器、n-gram重叠度样例、相似度样例选择器

from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

example = [
    {
        "question": "谁的寿命更长，悟空还是如来",
        "answer":
            """
            这里需要跟进问题吗： 是的。
            跟进： 悟空在记载中，最后能够追溯到他有多大年龄？
            中间答案： 五千岁
            跟进： 如来呢？
            中间答案： 佛祖在记载中，最后能够追溯到他多大年龄是一万岁。
            所以最终答案是： 悟空五千年，如来一万岁，所以如来寿命更长。
            """
    },
    {
        "question": "理论重要还是实践重要",
        "answer":
            """
            这里需要跟进问题吗： 是的。
            跟进：理论重要吗？
            中间答案： 理论当然重要，理论为实践提供了指导和框架，帮助我们理解和预测事物的规律。
            跟进： 实践重要吗？
            中间答案： 实践则是理论的验证和应用，通过实践，我们能够检验和调整理论，发现新的问题和需求。
            所以最终答案是： 都很重要。
            """
    },
]

# 提示词模板
example_prompt = PromptTemplate.from_template(template="问题：{question}\n{answer}")
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=example,
    suffix="问题：{input}",
    input_variables=["input"]
)
print(prompt.format(input="谁更厉害，孙悟空还是如来"))


from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

class DeepSeekEmbeddings:
    def __init__(self, model_name="deepseek-local"):
        self.model_name = model_name
        self.embedding_dim = 1024  # 必须与实际模型维度一致

    def embed_documents(self, texts):
        # 这里只是一个简单的自定义 text to float
        # text to vector, 应该需要 可以用 word2vec, Bert，，，
        return [
            [float(i%100)/100 for i in range(self.embedding_dim)]
            for _ in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    example,
    DeepSeekEmbeddings(),
    Chroma,
    k=1  # 选择相似样本的数量
)

question = "寿命长短"
selected_examples = example_selector.select_examples({"question": question})
print("question:", question)
print(f"最相似的示例:")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")

