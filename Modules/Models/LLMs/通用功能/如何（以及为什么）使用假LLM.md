# 如何（以及为什么）使用假LLM

我们公开了一个可以用于测试的假LLM类。这允许您模拟对LLM的调用，并模拟如果LLM以某种方式响应会发生什么。

在本笔记中，我们将学习如何使用它。

我们首先在代理中使用FakeLLM。

```python
from langchain.llms.fake import FakeListLLM

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

tools = load_tools(["python_repl"])
responses=[
    "Action: Python REPL\nAction Input: print(2 + 2)",
    "Final Answer: 4"
]
llm = FakeListLLM(responses=responses)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("whats 2 + 2")

```

```pycon
> Entering new AgentExecutor chain...
Action: Python REPL
Action Input: print(2 + 2)
Observation: 4

Thought:Final Answer: 4

> Finished chain.
```

```pycon
'4'
```