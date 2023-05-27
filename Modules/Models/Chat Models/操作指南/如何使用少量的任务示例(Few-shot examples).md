# 如何使用少量的任务示例(Few-shot examples)

这个笔记涵盖了如何在聊天模型中使用一些少量的任务示例(Few-shot examples)。

对于如何最好地进行Few-shot examples 提示，似乎没有达成一致意见。因此，我们还没有固化任何抽象，而是使用现有的抽象。

## 交替的人类/AI消息

第一种方法是使用交替的人工/AI信息来进行Few-shot examples 提示。请参见下面的示例。

```python
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
```

```python
chat = ChatOpenAI(temperature=0)
```

```python
template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
```
```pycon
"I be lovin' programmin', me hearty!"
```

## System Messages
OpenAI提供了一个可选参数`name`，他们还建议将其与系统消息结合使用，以进行Few-shot prompting。下面是如何做到这一点的一个例子。

```python
template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = SystemMessagePromptTemplate.from_template("Hi", additional_kwargs={"name": "example_user"})
example_ai = SystemMessagePromptTemplate.from_template("Argh me mateys", additional_kwargs={"name": "example_assistant"})
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
```
```python
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
# get a chat completion from the formatted messages
chain.run("I love programming.")
```

```pycon
"I be lovin' programmin', me hearty."
```