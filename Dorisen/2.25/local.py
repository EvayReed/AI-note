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
# 😴🤔👩‍💻
# <think>
# 嗯，用户让我用三个emoji来描述程序员的工作日常。首先，我得想一下程序员的工作内容和环境。程序员通常在办公室工作，负责编写代码、处理数据，有时候还要解决技术问题。所以，三个 emoji应该能很好地代表这些元素。
#
# 第一个emoji应该是用来表示编程的，比如“ coding”或者“ code”，但可能更偏向视觉上的思考模式。第二个可以是“办公桌”或者“ laptop”，这样能反映程序员在电脑前工作的ugging”有点长，可能不太合适。再想想，是否有更简洁或更有节奏感的选项？
#
# 哦，对了，“ coding”可以是第一个，办公室里的电脑，第二个是办公桌，第三个可能是“思维模式”或者“思考。”不过“思考”可能有点抽象，不太具体。或许可以用“专注”来表序员的认真态度？但那样的话三个emoji就变成了“coding”，“办公桌”，“专注”，这样感觉有点重复了。
#
# 再想一想，是否有更贴切的选项。比如，“编程”、“代码”、“键盘”？或者“办公桌”、“键盘”、“问题解决”。或许“编程”和“办公桌”已经足够好了，第三个可以用“思考模式”或者“专注状态”，但可能不够生动。或者用“思维模式”来描述程序员的工作态度。
#
# 再考虑一下，是否可以选择“编程”作为第一个，然后是“办公室”，接着是“思考过程”。“办公室”可以代表工作环境，“思考过程”则突出了程序员在解决问题时的注意力点。这样组合起来，三个emoji就能很好地表达程序员工作的日常了。
# </think>
#
# coding office thinking

