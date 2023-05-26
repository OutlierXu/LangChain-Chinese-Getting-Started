# LangChain-Chinese-Getting-Started
#  快速入门指南 

本教程让您快速了解如何使用 LangChain 构建端到端语言模型应用程序。

## 安装



首先，使用以下命令安装LangChain：

```python
pip install langchain
# or
conda install langchain -c conda-forge
```



## 环境设置 

使用LangChain通常需要与一个或多个模型提供程序，数据存储，API等集成。
在这个例子中，我们将使用OpenAI的API，所以我们首先需要安装他们的SDK：
```python
pip install openai
```
然后我们需要在终端中设置环境变量。
```python
export OPENAI_API_KEY="..."
```
或者，你可以在Jupyter笔记本（或Python脚本）中这样做：
```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```
如果您想动态设置API密钥，可以在初始化OpenAI类时使用openai_api_key参数，例如，每个用户的API密钥。
```python
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="OPENAI_API_KEY")
```

## 构建语言模型应用程序：LLM 
现在我们已经安装了LangChain并设置了我们的环境，我们可以开始构建语言模型应用程序了。

LangChain提供了许多模块，可用于构建语言模型应用程序。模块可以组合起来创建更复杂的应用程序，也可以单独用于简单的应用程序。

## LLM：从语言模型获取预测

LangChain最基本的构建块是在一些输入上调用LLM。让我们通过一个简单的例子来说明如何做到这一点。为此，让我们假设我们正在构建一个服务，该服务根据公司的生产情况生成公司名称。

为了做到这一点，我们首先需要导入LLM包装器。

```python
from langchain.llms import OpenAI
```

然后我们可以用任何参数初始化包装器。在这个例子中，我们可能希望输出更随机，所以我们将初始化高的temperature值。

```python
llm = OpenAI(temperature=0.9)
```
我们现在可以加一些输入进行调用！
```python
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
```
```python
Feetful of Fun
```
有关如何在LangChain中使用LLM的更多详细信息，请参阅LLM入门指南。



## Prompt模板：管理LLMs的prompts

调用LLM是很好的第一步，但这只是开始。通常，当您在应用程序中使用LLM时，您不会直接将用户输入发送到LLM。相反，您可能会接受用户输入并构造一个prompts，然后将其发送给LLM。

例如，在前面的示例中，我们传入的文本被硬编码，要求提供生产彩色袜子的公司的名称。在这个假想的服务中，我们想要做的是只接受描述公司做什么的用户输入，然后用这些信息格式化提示。

使用LangChain很容易做到这一点！

首先定义prompts模板：

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

现在让我们看看这是如何工作的！我们可以调用 .format 方法来格式化它。

```python
print(prompt.format(product="colorful socks"))
```
```python
What is a good name for a company that makes colorful socks?
```
[有关更多详细信息，请查看入门指南中的提示。](../modules/prompts/chat_prompt_template.ipynb)

## 链：在多步骤工作流中组合LLM和prompts

到目前为止，我们已经单独使用了PromptTemplate和LLM原语。当然，真实的的应用程序不仅仅是一个原语，而是它们的组合。

LangChain中的链由链接组成，链接可以是LLM之类的原语，也可以是其他链。

链最核心的类型是LLMCChain，它由PromptTemplate和LLM组成。

扩展前面的示例，我们可以构造一个LLMChain，它接受用户输入，使用PromptTemplate对其进行格式化，然后将格式化的响应传递给LLM。

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```
我们现在可以创建一个非常简单的链，它将接受用户输入，用它格式化prompt，然后将其发送给LLM：
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```
现在我们可以指定产品来运行该链了！
```pycon
chain.run("colorful socks")
# -> '\n\nSocktastic!'
```
好了！这是第一条链-一条LLM链。这是比较简单的链类型之一，但是理解它是如何工作的将使您能够很好地处理更复杂的链。

[有关更多详细信息，请查看链条入门指南。](../modules/chains/getting_started.ipynb)

## 代理：基于用户输入的动态调用链 

到目前为止，我们所看到的链都是按照预定的顺序运行的。

代理不再执行以下操作：他们使用LLM来确定采取哪些动作以及以什么顺序。动作可以是使用工具并观察其输出，也可以是返回给用户。

如果使用得当，代理可以非常强大。在本教程中，我们将向您展示如何通过最简单、最高级的API轻松使用代理。

为了加载代理，您应该了解以下概念：

- 工具：执行特定职责的函数。这可以是如下内容：Google搜索，数据库查找，Python REPL，其他链。工具的接口目前是一个函数，预期该函数将字符串作为输入，并将字符串作为输出。
- LLM：为代理提供动力的语言模型。
- 代理：要使用的代理。这应该是引用支持代理类的字符串。由于本笔记本侧重于最简单、最高级的API，因此仅涵盖了使用标准支持的代理。

如果要实现自定义代理，请参阅自定义代理的文档（即将推出）。

**代理**:有关支持的代理及其规格的列表，请参阅 [此处](../modules/agents/getting_started.ipynb).

**工具**: 有关预定义工具及其规范的列表，请参阅 [此处](../modules/agents/tools/getting_started.md).

对于本例，您还需要安装SerpAPI Python包。

```bash
pip install google-search-results
```

并设置适当的环境变量。

```python
import os
os.environ["SERPAPI_API_KEY"] = "..."
```
现在我们可以开始了！
```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
```
```pycon
> Entering new AgentExecutor chain...
 I need to find the temperature first, then use the calculator to raise it to the .023 power.
Action: Search
Action Input: "High temperature in SF yesterday"
Observation: San Francisco Temperature Yesterday. Maximum temperature yesterday: 57 °F (at 1:56 pm) Minimum temperature yesterday: 49 °F (at 1:56 am) Average temperature ...
Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
Action: Calculator
Action Input: 57^.023
Observation: Answer: 1.0974509573251117

Thought: I now know the final answer
Final Answer: The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

> Finished chain.
```

## 内存：将状态添加到链和代理 
到目前为止，我们所经历的所有链和代理都是无状态的。但通常情况下，您可能希望链或代理具有某种“记忆”的概念，以便它可以记住有关其先前交互的信息。

最清晰和简单的例子是在设计聊天机器人时-你希望它记住以前的消息，以便它可以使用上下文来进行更好的对话。这是一种“短期记忆”。

在更复杂的方面，你可以想象一个链/代理随着时间的推移记住关键的信息片段-这将是一种形式的“长期记忆”。关于后者的更具体的想法，请参阅这篇令人[敬畏的论文](https://memprompt.com/)。

LangChain为此提供了几个专门创建的链。这个笔记本使用其中一个具有两种不同类型内存的链（`ConversationChain`）进行了演示。

默认情况下，  `ConversationChain` 有一个简单类型的内存，它可以记住所有以前的输入/输出，并将它们添加到传递的上下文中。让我们来看看如何使用这个链（设置 `verbose=True` ，这样我们就可以看到提示符）。

```python
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)
```
```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished chain.
' Hello! How are you today?'
```
```python
output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)
```
```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:  Hello! How are you today?
Human: I'm doing well! Just having a conversation with an AI.
AI:

> Finished chain.
" That's great! What would you like to talk about?"
```

## 构建语言模型应用程序：聊天模型 
同样，您可以使用聊天模型而不是LLM。聊天模型是语言模型的变体。

虽然聊天模型在引擎盖下使用语言模型，但它们暴露的界面有点不同：它们不是公开“文本输入，文本输出”API，而是公开一个接口，其中“聊天消息”是输入和输出。

聊天模型API是相当新的，所以我们仍在找出正确的抽象。

## 从聊天模型获取消息完成 
您可以通过向聊天模型传递一条或多条消息来完成聊天。回复将是一条消息。LangChain目前支持的消息类型有 `AIMessage` 、 `HumanMessage` 、 `SystemMessage` 、 `ChatMessage` - `ChatMessage` ，可在任意角色参数中获取。大多数时候，你只需要处理 `HumanMessage` 、 `AIMessage` 和 `SystemMessage` 。
```pycon
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
```
你可以通过传递一条消息来获取completions 。
```pycon
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

您还可以为OpenAI的gpt-3.5-turbo和gpt-4模型传递多条消息。

```pycon
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

您可以更进一步，使用 `generate` 为多组消息生成补全。这将返回一个带有额外 `message` 参数的 `LLMResult` ：
```pycon
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result
# -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}})
```
你可以从这个LLMResult中获取信息，例如token使用情况：
```pycon
result.llm_output['token_usage']
# -> {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}
```
##  聊天提示模板 
与LLM类似，您可以通过使用 `MessagePromptTemplate` 来使用模板。您可以从一个或多个 `MessagePromptTemplate` 构建 `ChatPromptTemplate` 。您可以使用 `ChatPromptTemplate` 的 `format_prompt` -这将返回一个 `PromptValue` ，您可以将其转换为字符串或 `Message` 对象，这取决于您是否希望使用格式化的值作为llm或聊天模型的输入。

为了方便起见，模板上公开了一个 `from_template` 方法。如果你使用这个模板，它看起来会是这样的：
```pycon
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

## 链与聊天模型 
上一节中讨论的 `LLMChain` 也可以与聊天模型一起使用：

```pycon
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="French", text="I love programming.")
# -> "J'aime programmer."
```


## 带有聊天模型的代理

代理也可以与聊天模型一起使用，您可以使用 `AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION` 作为代理类型来初始化一个。
```pycon
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
chat = ChatOpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
```
```pycon

> Entering new AgentExecutor chain...
Thought: I need to use a search engine to find Olivia Wilde's boyfriend and a calculator to raise his age to the 0.23 power.
Action:
{
  "action": "Search",
  "action_input": "Olivia Wilde boyfriend"
}

Observation: Sudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.
Thought:I need to use a search engine to find Harry Styles' current age.
Action:
{
  "action": "Search",
  "action_input": "Harry Styles age"
}

Observation: 29 years
Thought:Now I need to calculate 29 raised to the 0.23 power.
Action:
{
  "action": "Calculator",
  "action_input": "29^0.23"
}

Observation: Answer: 2.169459462491557

Thought:I now know the final answer.
Final Answer: 2.169459462491557

> Finished chain.
'2.169459462491557'
```
## 内存：将状态添加到链和代理 
您可以将Memory与使用聊天模型初始化的链和代理一起使用。这与Memory for LLM之间的主要区别在于，我们可以将它们作为自己的唯一内存对象，而不是试图将所有以前的消息压缩为字符串。
```pycon
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

conversation.predict(input="Hi there!")
# -> 'Hello! How can I assist you today?'


conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
# -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

conversation.predict(input="Tell me about yourself.")
# -> "Sure! I am an AI language model created by OpenAI. I was trained on a large dataset of text from the internet, which allows me to understand and generate human-like language. I can answer questions, provide information, and even have conversations like this one. Is there anything else you'd like to know about me?"
```
