# 自定义LLM代理（带ChatModel）

这个笔记本讲述了如何基于聊天模型创建自己的定制代理。
LLM聊天代理由三部分组成：
- PromptTemplate：这是提示模板，可用于指示语言模型做什么;
- ChatModel：这是为代理提供动力的语言模型;
- `stop` 序列：指示LLM在找到此字符串后立即停止生成;
- `OutputParser`：这决定了如何将LLMOutput解析为AgentAction或AgentFinish对象;

LLMAgent用于AgentExecutor。这个AgentExecutor在很大程度上可以被认为是一个循环：
1. 将用户输入和之前的任何步骤传递给代理（在本例中为LLMAgent）;
2. 如果代理返回 `AgentFinish` ，则将其直接返回给用户;
3. 如果代理返回 `AgentAction` ，则使用它调用工具并获取 `Observation`;
4. 重复上述步骤，将 `AgentAction` 和 `Observation` 传回代理，直到发出 `AgentFinish` 。

`AgentAction` 是由 `action` 和 `action_input` 组成的响应。 `action` 是指要使用的工具， `action_input` 是指该工具的输入。 `log` 也可以作为更多的上下文提供（可以用于日志记录，跟踪等）。

`AgentFinish` 是一个响应，包含要发送回用户的最终消息。这应用于结束代理运行。


在本笔记中，我们将介绍如何创建自定义LLM代理。


## 设置环境
做必要的导入等。
```bash
!pip install langchain
!pip install google-search-results
!pip install openai
```
```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from getpass import getpass
```

## 设置工具
设置agent可能要使用的任何工具。这可能是必要的，以便在prompt中输入（以便代理知道使用这些工具）。

```python
SERPAPI_API_KEY = getpass()
```

```python
# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]
```

## Prompt模版
Prompt模版指示代理要做什么。通常，模板应包括：
- `tools` ：代理有权访问哪些工具，以及如何以及何时调用这些工具。
- `intermediate_steps` ：这些是先前（ `AgentAction` ， `Observation` ）对的元组。它们通常不会直接传递给模型，而是由提示模板以特定的方式格式化。
- `input`: 用户输入
```python
# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""
```
```python
# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
```
```python
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
```

## Output Parser
输出解析器负责将LLM输出解析成 `AgentAction` 和 `AgentFinish` 。这通常在很大程度上取决于所使用的prompt。
这是您可以更改解析以执行重试、处理空白等的地方

```python
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
```
```python
output_parser = CustomOutputParser()
```

## 设置LLM
选择您想要使用的LLM！
```python
OPENAI_API_KEY = getpass()


llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
```

## 定义停止顺序
这很重要，因为它告诉LLM何时停止生成。

这在很大程度上取决于您使用的提示符和模型。通常，您希望它是您在提示符中使用的任何标记，以表示 `Observation` 的开始（否则，LLM可能会为您产生幻觉）。

## 设置代理
我们现在可以把所有的东西结合起来来建立我们的代理

```python
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
```

```python
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
```
## 使用代理

现在我们可以使用它了！

```python
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
```

```python
agent_executor.run("Search for Leo DiCaprio's girlfriend on the internet.")
```
```python
> Entering new AgentExecutor chain...
Thought: I should use a reliable search engine to get accurate information.
Action: Search
Action Input: "Leo DiCaprio girlfriend"

Observation:He went on to date Gisele Bündchen, Bar Refaeli, Blake Lively, Toni Garrn and Nina Agdal, among others, before finally settling down with current girlfriend Camila Morrone, who is 23 years his junior.
I have found the answer to the question.
Final Answer: Leo DiCaprio's current girlfriend is Camila Morrone.

> Finished chain.
```
```python
"Leo DiCaprio's current girlfriend is Camila Morrone."
```