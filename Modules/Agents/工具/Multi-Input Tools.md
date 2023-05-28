# Multi-Input Tools


这个笔记展示了如何使用一个需要多个输入的工具。推荐的方法是使用 `StructuredTool` 类。

```python
import os
os.environ["LANGCHAIN_TRACING"] = "true"
```

```python
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

llm = OpenAI(temperature=0)
```

```python
from langchain.tools import StructuredTool

def multiplier(a: float, b: float) -> float:
    """Multiply the provided floats."""
    return a * b

tool = StructuredTool.from_function(multiplier)
```

```python
# Structured tools are compatible with the STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION agent type. 
agent_executor = initialize_agent([tool], llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

```python
agent_executor.run("What is 3 times 4")
```

```python
> Entering new AgentExecutor chain...

Thought: I need to multiply 3 and 4
Action:
```
{
  "action": "multiplier",
  "action_input": {"a": 3, "b": 4}
}
```

Observation: 12
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "3 times 4 is 12"
}
```

> Finished chain.
```

```pycon
'3 times 4 is 12'
```

## 字符串格式的多输入工具
结构化工具的另一种选择是使用常规的 `Tool` 类并接受单个字符串。然后，该工具必须处理解析逻辑以从文本中提取相关值，这将工具表示与代理提示紧密耦合。如果底层语言模型不能可靠地生成结构化模式，这仍然是有用的。

让我们以乘法函数为例。为了使用它，我们将告诉代理生成“Action Input”作为长度为2的逗号分隔列表。然后，我们将编写一个精简的包装器，它接受一个字符串，将其分成两部分，围绕一个逗号，并将解析后的两边作为整数传递给乘法函数。


```python
def multiplier(a, b):
    return a * b

def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))
```

```python
llm = OpenAI(temperature=0)
tools = [
    Tool(
        name = "Multiplier",
        func=parsing_multiplier,
        description="useful for when you need to multiply two numbers together. The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2."
    )
]
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

```python
mrkl.run("What is 3 times 4")
```

```python
> Entering new AgentExecutor chain...
 I need to multiply two numbers
Action: Multiplier
Action Input: 3,4
Observation: 12
Thought: I now know the final answer
Final Answer: 3 times 4 is 12

> Finished chain.
```

```pycon
'3 times 4 is 12'
```