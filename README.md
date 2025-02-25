[官方文档地址](https://python.langchain.com/api_reference/index.html)

# 安装langchain

# 开始第一个demo

+ 创建一个项目

+ 在项目中运行

  ```shell
  pip install langchain 
  pip install langchain_openai
  ```

+ 在根目录创建.env文件

  ```json
  OPENAI_API_KEY=key
  ```

  ```shell
  pip install python-dotenv
  ```
  加载环境变量关键代码：

  ```python
  import os
  from dotenv import load_dotenv
  
  load_dotenv()  # 加载.env文件
  ```

  

+ 创建main.py

  ```python
  import os
  from dotenv import load_dotenv
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  load_dotenv()
  api_key = os.getenv("OPENAI_API_KEY")
  prompt = ChatPromptTemplate.from_template(
      "用三个emoji描述：{item}"
  )
  model = ChatOpenAI(openai_api_key=api_key)
  chain = prompt | model
  
  response = chain.invoke({"item": "程序员的工作日常"})
  print(response.content)
  ```

# 嵌入本地deepseek大模型

首先下载安装好ollama

```
ollama -v   
```

安装个最小版的deepseek

```
ollama run deepseek-r1:1.5b
```

新建local.py文件

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_template(
    "用三个emoji描述：{item}"
)

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)
chain = prompt | llm

response = chain.invoke({"item": "程序员的工作日常"})
print(response.content)
```

# LLM和ChatModule

二者的选择取决于任务是否需要结构化对话管理。若涉及多轮交互，应优先使用 ChatOllama；若仅为单次文本生成，OllamaLLM 则更为轻量高效

- **ChatOllama**
  输出结果为包含元数据的消息对象（如`content`属性存储文本），便于在 LangChain 的对话链（ConversationChain）中传递和解析。例如，调用`ai_msg.content`可提取回复文本，同时保留消息类型信息以实现对话流管理。

- **OllamaLLM**
  输出直接为字符串文本，适合与 LangChain 的提示模板（`PromptTemplate`）或处理管道（Pipeline）结合，快速构建文本生成任务的工作流。例如，在 RAG 应用中，可通过`chain = prompt | model`将提示词与模型串联，简化流程编排。

  

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaLLM
OllamaChat = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

Ollama_LLM = OllamaLLM(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

print(OllamaChat.invoke([
    AIMessage(role="system", content="你好，我是丁凯乐"),
    HumanMessage(role="user", content="你好，我是凯南"),
    AIMessage(role="system", content="很高兴认识你"),
    HumanMessage(role="system", content="你知道我叫什么吗？")

]))
print(Ollama_LLM.invoke("你好啊，你叫什么名字？"))

```

输出结果：

```shell
content='<think>\n\n</think>\n\n您好！我在中国，是通过中国的官方渠道和机构联系的。如果您有任何问题或需要帮助，请随时告诉我。我会尽力为您提供及时、准确的信息和帮助。' additional_kwargs={} response_metadata={'model': 'deepseek-r1:1.5b', 'created_at': '2025-02-25T06:53:09.443872Z', 'done': True, 'done_reason': 'stop', 'total_duration': 562442042, 'load_duration': 30472792, 'prompt_eval_count': 28, 'prompt_eval_duration': 97000000, 'eval_count': 40, 'eval_duration': 433000000, 'message': Message(role='assistant', content='', images=None, tool_calls=None)} id='run-be227598-7a92-4413-b636-9657b314cb95-0' usage_metadata={'input_tokens': 28, 'output_tokens': 40, 'total_tokens': 68}
<think>

</think>

您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。我的名称是DeepSeek-R1，我擅长通过思考来帮您解答复杂的数学，代码和逻辑推理等理工类问题。

Process finished with exit code 0
```



# 流式输出

```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

for chunk in llm.stream("胸有成竹是什么意思"):
    print(chunk, end="", flush=False)
```

flush置为False是为了不清空控制台



# 统计token的消耗

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434",
    stream_usage=True  # 流模式需开启统计
)

response = llm.invoke([
    HumanMessage(content="望梅止渴是什么意思")
])

# 输出内容和token统计
print(f"响应内容: {response.content}")
print(f"输入Token: {response.usage_metadata['input_tokens']}")
print(f"输出Token: {response.usage_metadata['output_tokens']}")
print(f"总消耗Token: {response.usage_metadata['total_tokens']}")
```



# 底层原理

## 特性

+ **LLM和提示（Prompt）**: LangChain 对大模型进行了API抽象，统一了大模型访问API, 同时提供了Prompt提示模板管理机制

+ **链（Chain）：** 链式调用，一步一步的执行

+ **LCEL**:  解决工作流编排问题，通过LCEL表达式，灵活的自定义AI任务处理流程，灵活自定义链

+ **数据增强生成（RAG）：**因为预训练时有截止时间范围的，大模型不了解新的信息，无法回答新的问题，所以我们可以将新的信息导入LLM， 用于增强LLM生成内容的质量；也就是推理。

  除了RAG的方式以外，还有一种途径叫微调

+ **Agents:** 根据用户需求自动调用外部系统，设备共同完成任务；例如用户输入：“明天请假一天” ， 大模型（LLM）自动调用请假系统，发起一个请假申请。

+ **模型记忆（memory）：**让大模型记住之前的对话内容；把内容存到数据库，在后面接入大模型的过程中，把记忆调出来；



## LangChain生态

+ **LangSmith**：链的追踪，相当于运维这一块；可观测性的监控系统，可以了解链的调用你情况

+ **LangServe:** 做部署的；langchain开发的应用；对外暴露REST API，可以通过http去请求langchain的中的API

+ **Templates：**模板的定义的

+ **LangChain**：

  + **Chains**： 链式调用

  + **Agents**：智能体

  + **Retrieval Srategies**: 向量数据库的检索的策略；

  + **LangChain-Community**: 

    一下都是开源的社区力量在支持的

    + **Model I/O**
      + Model： 模型参数
      + Prompt： 提示词模板
      + Example Selector： 样例，问大模型的时候提供的示例，聊天的时候会把示例带上去。
      + Output Parser： 格式化，默认是以markdown的形式
    + **Retrieval**（针对RAG组件的设计）
      + Retriever：向量数据库的检索
      + Document Loader：文档加载，给大模型喂数据；把加载的数据转换成向量
      + Vector Store：向量数据库
      + Text Splitter： 文本分割；大文件分片存储
      + Embedding Model：调用大模型的能力去实现的
    + **Agent Tooling**
      + Tool： 
      + Toolkit

  + **LangChain-Core**

    + **LCEL**： 表达式语言，可以让你用一套语法去很方便的做一个
    + **Parallelization**：并行调用
    + **Fallbacks**： 处理可能出现的异常或错误
    + **Tracing**： 跟踪
    + **Batching**：批量调用
    + **Streaming**： 流式调用
    + **Async**：异步调用
    + **Composition**： 组合调用



# LangChain结构组成

+ **LangChain库：**Python和JavaScript库，（用python居多）包含接口和集成多种组件的运行时基础以及现成的链和代理的实现
+ **Langchain模板**： LangChain官方提供的一些AI任务模板，执行AI任务的时候可以使用官方提供的模板库中的模板
+ **LangServe**： 基于FastAPI可以将LangChain定义的Chain发布成REST API
+ **LangSmith**： 开发平台，是个云服务，支持LangChain debug、任务监控；任务耗时，任务报错都可以看到



# LangChain Libraries

+ Langchain-core： 核心库；基础抽象和LangChain表达语言。
+ Langchain-comunity: 第三方的一些集成。主要包括langchain集成的第三方组件
+ langchain：基础的语言知识，主要包括chain、 agent和检索策略



# LangCchain任务处理流程

输入问题，经过提示词模板加工，以为LLM或者Chat Model的形式去与大模型交流；拿到结果后，通过output parse转换成想要的格式 

langchain提供一套提示词模板（prompt template）管理工具，负责处理提示词，然后传递给大模型处理，最后处理大模型返回的结果

Langchain对大模型的封装主要包括LLM和Chat Model两种类型

​	**LLM - 问答模型**： 模型接受一个文本输入，然后返回一个文本结果

​	**Chat Model -对话模型**：接受一组对话消息上下文（包括给AI的角色定义），然后返回对话消息。



## 核心概念

+ **LLMs：** 大模型，接收一个文本输入，然后返回一个文本结果
+ **Chat Models**： 聊天模型，与LLMs不同，这些模型专为对话场景而设计。
+ **Message：**指的是Chat Models的消息内容， 消息类型包括HumanMessage  AIMessage、 SystemMessage、 FunctionMessage和ToolMessage等多种类型消息
+ **prompts：** 提示词管理工具类，方便我们格式化提示词propts内容
+ **Output Parsers：**输出解析器,对大模型输出的内容进行自定义的输出
+ **Retrievers:**检索框架， 方便我们加载文档数据、切割文档数据、存储和检索文档数据
+ **Vector stores：** 支持私有素具的语义相似搜索；向量数据库例如faiss、chroma
+ **Agent：**智能体，以大模型作为决策，根据用户输入的任务，自动调用外部系统、硬件设备共同完成用户的任务，是一种以大模型LLM为核心的应用设计模式



# LangChain提示词模版

## PromptTemplate 消息模板

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("你的名字叫做{name},你的年龄是{age}")

result = prompt.format(name="张三", age=18)

print(result)
```

## ChatPromptTemplate  对话模板

+ **from_messages**

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个程序员,你的名字叫{name}"),
    ("human", "你好"),
    ("ai", "你好,我叫{name}"),
    ("human", "{user_input}"),
])
message = prompt_template.format_messages(name="张三", user_input="今天天气怎么样？")
print(message)
```

+ **from_template**

```python
prompt_template = ChatPromptTemplate.from_template(
    "你好，我是{name}，请问{user_input}"
)
message = prompt_template.format_messages(name="张三", user_input="今天天气怎么样？")
print(message)
```

Chat Model 聊天模式以聊天消息列表作为输入，每个消息都与角色 role想关联

三种角色分别是：

+ 助手（Assistant）消息指的是当前消息是AI回答的内容
+ 人类（user）消息指的是你发给AI的内容
+ 系统（system）消息通常是用来AI身份进行描述的



### 对象的方式进行定义

如果需要接收参数，就用SystemMessagePromptTemplate.from_template； 不需要接收参数就用SystemMessage，因为SystemMessage中的content是一个字符串而非模板字符串；其他两种类亦然

SystemMessage  系统；  HumanMessage 人类；  AIMessage 大模型

```python
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            "你是一个搞笑风趣的智能助手"
        )
    ),
    SystemMessagePromptTemplate.from_template(
        "你的名字叫做{name},请根据用户的问题回答。"
    ),
    HumanMessagePromptTemplate.from_template(
        "用户的问题是：{user_input}"
    ),
])
message = chat_template.format_messages(name="张三", user_input="今天天气怎么样？")
print(message)
```

三个类也是可以独立写，直接创建消息

```puthon
sy = SystemMessage(
	content="你是一个起名大师",
  assitional_kwargs={"大师名字":"王麻子"}
)

hu = HumanMessage(
	constent="请问大师叫什么"
)

ai = AIMessage(
	content="我叫王麻子"
)

print([sy, hu, ai])
```

三个类的模板化示例：

```python
prompt = "我是来自{address},我深爱着我的家乡"

# chat_prompt = AIMessagePromptTemplate.from_template(template=prompt)
# chat_prompt = HumanMessagePromptTemplate.from_template(template=prompt)
# chat_prompt = SystemMessagePromptTemplate.from_template(template=prompt)
# 前三者都不需要指定角色，下面这个是需要指定角色的
chat_prompt = ChatMessagePromptTemplate.from_template(role="这里不仅局限于那三种角色", template=prompt)

message = chat_prompt.format(address="地球")

print(message)
```



## StringPromptTemplate自定义模板

在自定义模板的时候，创建的类需要实例化format方法

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import StringPromptTemplate


def hello_word(address):
    print("Hello, world!" + address)
    return f"Hello, {address}!"


PROMPT = """\
    You are a helpful assistant that answers questions based on the provided context.
    function name: {function_name}
    source code:
    {source_code}
    explain:
"""

import inspect  # 这个包可以根据函数名，获取到函数源代码


def get_source_code(function_name):
    # 获取源代码
    return inspect.getsource(function_name)


class CustmPrompt(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        source_code = get_source_code(kwargs["function_name"])

        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__, source_code=source_code
        )
        return prompt


a = CustmPrompt(input_variables=["function_name"])
pm = a.format(function_name=hello_word)

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

result = llm.invoke(pm)
print(result.content)

```

## 混合分层模版

使用PipelinePromptTemplate来把多层定义联合起来

```python
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

full_template = """{Character}
{Behavior}
{Prohibit}"""
full_prompt = PromptTemplate.from_template(full_template)

Character_template = "你是{person}, 你有着{attribute}"
Character_prompt = PromptTemplate.from_template(
    Character_template
)

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
```



## 序列化模板使用

创建一个yaml文件

```yaml
_type: prompt
input_variables:
  ["name", "what"]
template:
  给我讲一个关于{name}的{what}的故事
```

或者创建json文件

```json
{
  "_type": "prompt",
  "input_variables": ["name", "what"],
  "template": "给我讲一个关于{name}的{what}的故事"
}
```

使用load_prompt来加载文件

```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("./prompts/simple_prompt.yaml")
print(prompt.format(name="科学家", what="热爱搞发明"))
```



## MessagesPlaceholder

相当于一个消息的占位符，可以传入一组消息。  可以把一段聊天记录插进去，提供上下文。

```python
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个搞笑风趣的智能助手"),
    MessagesPlaceholder("msgs")
])
message = chat_template.invoke({"msgs": [HumanMessage(content="hi!"), HumanMessage(content="hello!")]})
print(message)
```

## 示例筛选

### LengthBasedExampleSelector

长度动态选择提示词组

可以配置LengthBasedExampleSelector中的max_length来决定生成的提示词长度，决定选择多少提示词组；

注意，dynamic_prompt.format(adjective="forward")   入参adjective的长度也会影响到选中的词组数量

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 假设已经有这么多的提示词示例组
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "rainy"},
    {"input": "windy", "output": "calm"},
    {"input": "hot", "output": "cold"},
    {"input": "fast", "output": "slow"},
    {"input": "big", "output": "small"},
    {"input": "bright", "output": "dark"},
    {"input": "strong", "output": "weak"},
    {"input": "clean", "output": "dirty"},
    {"input": "heavy", "output": "light"},
    {"input": "happy", "output": "angry"},
    {"input": "high", "output": "low"},
    {"input": "rich", "output": "poor"},
    {"input": "beautiful", "output": "ugly"},
    {"input": "full", "output": "empty"},
    {"input": "young", "output": "old"},
    {"input": "loud", "output": "quiet"},
    {"input": "soft", "output": "hard"},
    {"input": "strong", "output": "fragile"},
    {"input": "sweet", "output": "sour"},
    {"input": "clean", "output": "messy"},
    {"input": "open", "output": "closed"},
    {"input": "warm", "output": "cool"}
]

# 构造提示词模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n 反义：{output}"
)

# 调用长度示例选择器
example_selector = LengthBasedExampleSelector(
    examples=examples,  # 传入示例提示词组
    example_prompt=example_prompt,  # 传入的提示词模板
    max_length=20  # 格式化后，的提示词的最大长度
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义:",
    input_variables=["adjective"]
)

print(dynamic_prompt.format(adjective="forward"))

```

### MMR检索相关示例

MMR是一种在信息检索中常用的方法，它的目标是在相关性和多样性之间找到一个平衡

首先找出与输入相似（余弦相似度最大）的样本

比方要选出50个，并不是一次性选出50个的，而是一次选一个，直到选到50个

然后在迭代添加样本的过程中，对于与选择样本过于接近（即相似度过高）的样本进行惩罚（选择过的就不会再选了）

技能确保选出样本与输入高度相关，又能保证选出的样本之间有足够的多样性

关注如何在相关性和多样性之间找到一个平衡

```
pip install faiss-cpu
```

```python
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# 假设已经有这么多的提示词示例组
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "rainy"},
    {"input": "windy", "output": "calm"},
    {"input": "hot", "output": "cold"},
    {"input": "fast", "output": "slow"},
    {"input": "big", "output": "small"},
    {"input": "bright", "output": "dark"},
    {"input": "strong", "output": "weak"},
    {"input": "clean", "output": "dirty"},
    {"input": "heavy", "output": "light"},
    {"input": "happy", "output": "angry"},
    {"input": "高兴", "output": "悲伤"},
    {"input": "愤怒", "output": "冷静"},
    {"input": "晴天", "output": "雨天"},
    {"input": "有风", "output": "平静"},
    {"input": "轻松", "output": "紧张"},
    {"input": "快", "output": "慢"},
    {"input": "亲切", "output": "冷漠"},
    {"input": "明亮", "output": "暗"},
    {"input": "强", "output": "弱"}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n反义：{output}"
)

# 初始化OpenAIEmbeddings时显式传递密钥
# embeddings = OpenAIEmbeddings(openai_api_key=api_key)
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b"
)

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    embeddings,  # 使用已初始化的embeddings对象
    FAISS,
    k=2 # 选中的数量
)

mmr_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="请根据以下示例生成反义词：",
    suffix="原词是：{adjective}\n反义词：",
    input_variables=["adjective"]
)

print(mmr_prompt.format(adjective="worried"))
```

### 根据输入相似度选择实例

它通过计算两个向量（在这里，向量可以代表文本、句子或则词语）之间的余弦值来衡量他们的相似度

余弦值越接近1，表示两个向量越相似

主要关注的是如何准确衡量两个向量的相似度

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from langchain_community.vectorstores import Chroma

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,  # 使用已初始化的embeddings对象
    Chroma,
    k=2
)
```



## 提示词追加示例

**FewShotPromptTemplate**

可以有效的解决大模型出现幻觉的问题，如果没有给出示例，那么生成的内容就会比较宽泛. 

可以看作是一个简单的知识库，风格，格式，答案；如果在模版中答案已经有了的话，就会直接从答案中组织语言作答。

```python
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

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题：{question}\n{answer}")
# 做了个格式化

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,  # 这里需要是PromptTemplate(提示词模板)
    examples=example,  # 这里需要是列表
    suffix="问题：{input}",
    input_variables=["input"]
)

print(prompt.format(input="谁更厉害，孙悟空还是如来"))

```

也可以之塞入其中一个示例：

```python
print(example_prompt.format(**examples[0])
      
# 相当于：
      
print(example_prompt.format("question"="谁的寿命更长，悟空还是如来", "answer"=
            """
            这里需要跟进问题吗： 是的。
            跟进： 悟空在记载中，最后能够追溯到他有多大年龄？
            中间答案： 五千岁
            跟进： 如来呢？
            中间答案： 佛祖在记载中，最后能够追溯到他多大年龄是一万岁。
            所以最终答案是： 悟空五千年，如来一万岁，所以如来寿命更长。
            """)
```





## 示例选择器(向量相似度匹配)

会有成千上万个示例，为了节省token，不可能把所有的示例都带上。所以在选样本数据的时候，需要把问的问题跟示例上的问题做匹配，然后取出来

使用 **SemanticSimilarityExampleSelector**根据输入的相似性选择小样本示例，它使用嵌入式模型计算输入和小样本示例之间的相似性，然后使用向量数据库执行相似搜索，获取跟输入相似的示例。

```shell
pip install chromadb
```

```python
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(openai_api_key="..."),
    Chroma,
    k=1
)

question = "寿命长短"
selected_examples = example_selector.select_examples({"question": question})

print(f"最相似的示例:{question}")
for example in selected_examples:
    print("\\n")
    for k, v in example.items():
        print(f"{k}: {v}")
```

使用本地大模型

```python
class DeepSeekEmbeddings:
    def __init__(self, model_name="deepseek-local"):
        self.model_name = model_name
        self.embedding_dim = 1024  # 必须与实际模型维度一致

    def embed_documents(self, texts):
        return [
            [float(i%100)/100 for i in range(self.embedding_dim)]
            for _ in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]
```

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

class DeepSeekEmbeddings:
    def __init__(self, model_name="deepseek-local"):
        self.model_name = model_name
        self.embedding_dim = 1024  # 必须与实际模型维度一致

    def embed_documents(self, texts):
        return [
            [float(i%100)/100 for i in range(self.embedding_dim)]
            for _ in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    DeepSeekEmbeddings(),
    Chroma,
    k=1
)

question = "快乐星球在哪里"
selected_examples = example_selector.select_examples({"question": question})

print(f"最相似的示例:{question}")
for example in selected_examples:
    print("\n".join([f"{k}: {v}" for k, v in example.items()]))
```

# OutPut Parsers

自定义的输出规则，让LLM不仅只是聊天文本，还可以与现实各种系统无缝对接

## 输出函数参数

```python
pip install pydantic
```

[Pydantic](https://docs.pydantic.dev/dev/) 是一个基于Python的数据验证和设置管理库，通过使用Pydantic，可以轻松为数据模型设置类型注释，以确保数据的**有效性和正确性**。

# LangChain 工作流编排 

LCEL是一种强大的工作流编排工具，可以从基本组件构建复杂任务链条（chain）， 并支持诸如流式处理、并行处理和日志记录等开箱急用的功能

+ **流式调用**

使用 LCEL（LangChain 表达式语言）构建处理链时，最大的优势在于它能以最快的速度让用户看到第一个输出结果。例如，当你输入一个请求后，系统会像流水线一样逐块处理数据：语言模型生成一部分内容，解析器立刻处理这部分内容并实时显示给用户。整个过程类似于 “边生成边解析”，而非等所有内容生成完毕后再统一处理。因此，用户几乎能在语言模型生成文字的同时，逐步看到解析后的结果，无需长时间等待全部内容生成完成。这种设计特别适用于需要即时反馈的场景

+ **异步支持**

可以使用同步以及异步API进行调用，能够在同一服务器中处理许多并发请求；更关键的是，当流程中存在可并行执行的环节（如同时检索多个数据库），LCEL 会自动优化执行顺序，显著减少整体延迟。这种设计既保持了代码简洁性，又实现了专业级系统的性能要求。

+ **并行执行优化机制**

LCEL（LangChain 表达式语言）的并行执行优化机制旨在提升 AI 任务链的处理效率。其核心原理在于：当开发者构建的链式流程中存在多个可独立运行的步骤（例如同时调用多个文档检索器），系统会自动识别这些并行化节点，并在同步或异步接口中同时调度执行，而非按顺序逐个处理。这种动态并发控制能够显著降低任务链的整体延迟，尤其在涉及 I/O 密集型操作（如网络请求或外部数据检索）时，通过消除不必要的等待时间实现性能优化。

具体而言，该机制通过以下技术路径实现：

​	首先，在链式结构解析阶段，LCEL 会分析各节点的依赖关系，识别出无状态关联且资源占用不冲突的可并行模块；

​	其次，系统根据运行时环境自动选择最优调度策略，例如在同步接口中采用多线程并发，在异步接口中则利用协程非阻塞执行；

​	最后，所有并行节点的执行结果会被动态整合至后续处理流程，确保数据流的一致性。这种智能化的并行化处理使得开发者无需手动设计并发逻辑，即可在保持代码简洁性的同时获得接近理论极限的执行效率。

+ **重试和回退**

在调用失败的时候，会重试和回退

+ **访问中间结果**

对于更复杂的链，访问中间步骤的结果通常非常有用，即使在生成最终输出之前，这可以用于让最终用户知道正在发生的事情。并且在每个LangServe服务器上都可以使用

