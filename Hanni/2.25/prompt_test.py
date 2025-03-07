from langchain.prompts import PromptTemplate

# 提示模板的创建
# 方式1：
prompt = PromptTemplate(input_variables=['name', 'age'], template="你的名字叫做{name},你的年龄是{age}")
result = prompt.format(name="张三", age=18)
print(result)

# 方式2： 使用from_template 不用显式指定input_variables
prompt_template = PromptTemplate.from_template(template = "你的名字叫做{name},你的年龄是{age}")
result = prompt.format(name="张三", age=18)
print(result)



# PipelinePromptTemplate 管道提示模板  用于把几个提示组合在一起使用
from langchain_core.prompts import PipelinePromptTemplate
full_template = """{Character}
{Behavior}
{Prohibit}"""
full_prompt = PromptTemplate.from_template(full_template)

Character_template = "你是{person}, 你有着{attribute}"
Character_prompt = PromptTemplate.from_template(Character_template)

behavior_template = "你会遵从以下行为：\n{behavior_list}"
behavior_prompt = PromptTemplate.from_template(behavior_template)

prohibit_template = "你不能遵从以下行为：\n{prohibit_list}"
prohibit_prompt = PromptTemplate.from_template(prohibit_template)

input_prompts = [
    ("Character", Character_prompt),
    ("Behavior", behavior_prompt),
    ("Prohibit", prohibit_prompt),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts=input_prompts
)

print(pipeline_prompt.format(
    person="科学家",
    attribute="严谨",
    behavior_list="1. 确保实验结果准确\n2. 记录实验过程\n3. 分析实验数据",
    prohibit_list="1. 不做实验\n2. 不记录实验过程\n3. 不分析实验数据"
))




# 聊天提示模板(Chat prompt template)  聊天模式三种角色 System AI Human
from langchain.prompts import ChatPromptTemplate
# messageTemplate创建

# ChatPromptTemplate.from_messages 显式指定角色
prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个程序员,你的名字叫{name}"),
    ("human", "你好"),
    ("ai", "你好,我叫{name}"),
    ("human", "{user_input}"),
])
message = prompt_template.format_messages(name="张三", user_input="今天天气怎么样？")
print(message)


# 使用ChatPromptTemplate.from_template 会默认HumanMessage
chat_prompt = ChatPromptTemplate.from_template(template="你好，我是{name}，请问{user_input}")
result = chat_prompt.format_messages(name="张三", user_input="今天天气怎么样？")
print(result)


# 模板需要接收参数时使用：
from langchain.prompts import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# 不需要接收参数时：
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=("你是一个搞笑风趣的智能助手")),
    SystemMessagePromptTemplate.from_template("你的名字叫做{name},请根据用户的问题回答。"),
    HumanMessagePromptTemplate.from_template("用户的问题是：{user_input}"),
])
message = chat_template.format_messages(name="张三", user_input="今天天气怎么样？")
print(message)


# 三种角色 模板接收参数
template = "我是来自{address},我深爱着我的家乡"
chat1 = AIMessagePromptTemplate.from_template(template=template).format(address="地球")
chat2 = HumanMessagePromptTemplate.from_template(template=template).format(address="地球")
chat3 = SystemMessagePromptTemplate.from_template(template=template).format(address="地球")
print([chat1, chat2, chat3])


# ChatMessagePromptTemplate需要显式指定角色，不局限于System AI Human
from langchain_core.prompts import ChatMessagePromptTemplate
chat_prompt = ChatMessagePromptTemplate.from_template(role="这里不仅局限于那三种角色", template=template)
message = chat_prompt.format_messages(address="地球")
print(message)


# MessagesPlaceholder占位符
# 可以多个
from langchain_core.prompts import MessagesPlaceholder
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个搞笑风趣的智能助手"),
    MessagesPlaceholder("msgs"),
    MessagesPlaceholder("test")
])
message = chat_template.invoke({"msgs": [SystemMessage(content="hi!"),
                                         HumanMessage(content="hello!")],
                                "test": [AIMessagePromptTemplate.from_template(template='我是{name}').format(name="智能助手"),
                                         AIMessage(content="more!")]})

print(message)




# 样本示例 提示模板
examples = [
    {"input": "2+2", "output": "4", "description": "加法运算"},
    {"input": "5-2", "output": "3", "description": "减法运算"},
    {"input": "3+5", "output": "8", "description": "加法运算"},
    {"input": "7-2", "output": "5", "description": "减法运算"},
]
prompt_template = "你是一个数学专家. 算式:{input}, 结果值:{output}, 使用的是:{description} "
prompt_sample = PromptTemplate.from_template(template=prompt_template)
print(prompt_sample.format(**examples[0]))  # 你是一个数学专家. 算式:2+2, 结果值:4, 使用的是:加法运算
print(prompt_sample.format(**examples[1]))  # 你是一个数学专家. 算式:5-2, 结果值:3, 使用的是:减法运算

# 少量样本示例的提示模板  FewShotPromptTemplate 或 FewShotChatMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt_sample,
    suffix="你是一个数学专家. 算式:{input}, 结果值:{output}",
    input_variables=["input", "output"]
)
print(prompt.format(input="2*5", output="10"))  # 你是一个数学专家,算式: 2*5  值: 10


# 样例选择器
# 自定义、长度选样、MMR样例选择器、n-gram重叠度样例、相似度样例选择器

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

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

class DeepSeekEmbeddings:
    def __init__(self, model_name="deepseek-local"):
        self.model_name = model_name
        self.embedding_dim = 1024  # 必须与实际模型维度一致

    def embed_documents(self, texts):
        # 这里只是一个简单的自定义 text to float
        # text to vector, 应该可以用 word2vec, Bert，，，
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


