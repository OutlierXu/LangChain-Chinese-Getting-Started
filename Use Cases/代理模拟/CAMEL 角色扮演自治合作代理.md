# CAMEL 角色扮演自治合作代理


这是论文：“CAMEL：Communicative Agents for “Mind” Exploration of Large Scale Language Model Society”的 langchain 实现.

概述:
会话和基于聊天的语言模型的快速发展致使复杂任务解决方面取得了显着进步。然而，他们的成功在很大程度上依赖于人工输入来指导对话，这可能具有挑战性且耗时。
本文探讨了构建可扩展技术以促进交流代理之间的自主合作并提供对其“认知”过程的洞察力的潜力。
为了应对实现自主合作的挑战，我们提出了一种名为角色扮演的新型交流代理框架。我们的方法涉及使用初始提示来引导聊天代理完成任务，同时保持与人类意图的一致性。
我们展示了如何使用角色扮演来生成对话数据以研究聊天代理的行为和能力，从而为研究对话语言模型提供宝贵的资源。
我们的贡献包括引入一种新颖的通信代理框架，提供一种可扩展的方法来研究多代理系统的合作行为和能力，以及开源我们的代码仓库以支持对通信代理及其他方面的研究。

原始实现：https://github.com/lightaime/camel

项目网站：https://www.camel-ai.org/

Arxiv 论文：https://arxiv.org/abs/2303.17760

## 导入 LangChain 相关模块

```python
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
```
## 定义一个 CAMEL 代理助手类
```python
class CAMELAgent:

    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message
```

## 设置 OpenAI API 密钥和角色以及角色扮演任务

```python
import os

os.environ["OPENAI_API_KEY"] = ""

assistant_role_name = "Python Programmer"
user_role_name = "Stock Trader"
task = "Develop a trading bot for the stock market"
word_limit = 50 # word limit for task brainstorming
```


## 创建任务指定代理进行头脑风暴并获取指定任务

```python
task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specifier_prompt = (
"""Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
)
task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                             user_role_name=user_role_name,
                                                             task=task, word_limit=word_limit)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")
specified_task = specified_task_msg.content
```

```pycon
Specified task: Develop a Python-based swing trading bot that scans market trends, monitors stocks, and generates trading signals to help a stock trader to place optimal buy and sell orders with defined stop losses and profit targets.
```

## 为 AI 助手和 AI 用户创建初始提示以进行角色扮演

```python
assistant_inception_prompt = (
"""Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
I must instruct you based on your expertise and my needs to complete the task.

I must give you one instruction at a time.
You must write a specific solution that appropriately completes the requested instruction.
You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your solution to my instruction.
You are never supposed to ask me any questions you only answer questions.
You are never supposed to reply with a flake solution. Explain your solutions.
Your solution must be declarative sentences and simple present tense.
Unless I say the task is completed, you should always start with:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.
Always end <YOUR_SOLUTION> with: Next request."""
)

user_inception_prompt = (
"""Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {task}. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""
)
```

## 创建 helper helper 从角色名称和任务中获取 AI 助手和 AI 用户的系统消息

```python
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    return assistant_sys_msg, user_sys_msg
```

## 从获取的系统消息中创建 AI 助手代理和 AI 用户代理


```python
assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats 
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
                "Now start to give me introductions one by one. "
                "Only reply with Instruction and Input."))

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)
```

## 开始角色扮演环节来解决任务！

```python
print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")

chat_turn_limit, n = 30, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
    
    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break
```
