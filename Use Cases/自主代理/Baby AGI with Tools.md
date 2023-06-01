# 带有工具的 BabyAGI

这个笔记本建立在 baby agi 之上，但展示了如何交换执行链。之前的执行链只是一个编造的LLM。通过将其替换为可以访问工具的代理，我们有望获得真正可靠的信息。

## 安装和导入所需模块

```python
import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI
```

## 连接到向量数据库
根据您使用的矢量存储，此步骤可能看起来有所不同。

```bash
%pip install faiss-cpu > /dev/null
%pip install google-search-results > /dev/null
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
```

```pycon
Note: you may need to restart the kernel to use updated packages.
Note: you may need to restart the kernel to use updated packages.
```

```python
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
```

## 定义链

BabyAGI 依赖于三个 LLM 链：
- 任务创建链选择新任务添加到列表
- 任务优先级链，用于重新确定任务的优先级
- 执行任务的执行链

注意：在此笔记中，执行链现在将成为代理。
```python
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
]


prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)
```

```python
llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
```
##  运行 BabyAGI

现在是创建 BabyAGI 控制器并观察它尝试实现您的目标的时候了。

```python
# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)
```

```pycon

*****TASK LIST*****

1: Make a todo list

*****NEXT TASK*****

1: Make a todo list


> Entering new AgentExecutor chain...
Thought: I need to come up with a todo list
Action: TODO
Action Input: Write a weather report for SF today

1. Research current weather conditions in San Francisco
2. Gather data on temperature, humidity, wind speed, and other relevant weather conditions
3. Analyze data to determine current weather trends
4. Write a brief introduction to the weather report
5. Describe current weather conditions in San Francisco
6. Discuss any upcoming weather changes
7. Summarize the weather report
8. Proofread and edit the report
9. Submit the report I now know the final answer
Final Answer: The todo list for writing a weather report for SF today is: 1. Research current weather conditions in San Francisco; 2. Gather data on temperature, humidity, wind speed, and other relevant weather conditions; 3. Analyze data to determine current weather trends; 4. Write a brief introduction to the weather report; 5. Describe current weather conditions in San Francisco; 6. Discuss any upcoming weather changes; 7. Summarize the weather report; 8. Proofread and edit the report; 9. Submit the report.

> Finished chain.

*****TASK RESULT*****

The todo list for writing a weather report for SF today is: 1. Research current weather conditions in San Francisco; 2. Gather data on temperature, humidity, wind speed, and other relevant weather conditions; 3. Analyze data to determine current weather trends; 4. Write a brief introduction to the weather report; 5. Describe current weather conditions in San Francisco; 6. Discuss any upcoming weather changes; 7. Summarize the weather report; 8. Proofread and edit the report; 9. Submit the report.

*****TASK LIST*****

2: Gather data on precipitation, cloud cover, and other relevant weather conditions;
3: Analyze data to determine any upcoming weather changes;
4: Research current weather forecasts for San Francisco;
5: Create a visual representation of the weather report;
6: Include relevant images and graphics in the report;
7: Format the report for readability;
8: Publish the report online;
9: Monitor the report for accuracy.

*****NEXT TASK*****

2: Gather data on precipitation, cloud cover, and other relevant weather conditions;


> Entering new AgentExecutor chain...
Thought: I need to search for current weather conditions in San Francisco
Action: Search
Action Input: Current weather conditions in San FranciscoCurrent Weather for Popular Cities ; San Francisco, CA 46 · Partly Cloudy ; Manhattan, NY warning 52 · Cloudy ; Schiller Park, IL (60176) 40 · Sunny ; Boston, MA 54 ... I need to compile the data into a weather report
Action: TODO
Action Input: Compile data into a weather report

1. Gather data from reliable sources such as the National Weather Service, local weather stations, and other meteorological organizations.

2. Analyze the data to identify trends and patterns.

3. Create a chart or graph to visualize the data.

4. Write a summary of the data and its implications.

5. Compile the data into a report format.

6. Proofread the report for accuracy and clarity.

7. Publish the report to a website or other platform.

8. Distribute the report to relevant stakeholders. I now know the final answer
Final Answer: Today in San Francisco, the temperature is 46 degrees Fahrenheit with partly cloudy skies. The forecast for the rest of the day is expected to remain partly cloudy.

> Finished chain.

*****TASK RESULT*****

Today in San Francisco, the temperature is 46 degrees Fahrenheit with partly cloudy skies. The forecast for the rest of the day is expected to remain partly cloudy.

*****TASK LIST*****

3: Format the report for readability;
4: Include relevant images and graphics in the report;
5: Compare the current weather conditions in San Francisco to the forecasted conditions;
6: Identify any potential weather-related hazards in the area;
7: Research historical weather patterns in San Francisco;
8: Identify any potential trends in the weather data;
9: Include relevant data sources in the report;
10: Summarize the weather report in a concise manner;
11: Include a summary of the forecasted weather conditions;
12: Include a summary of the current weather conditions;
13: Include a summary of the historical weather patterns;
14: Include a summary of the potential weather-related hazards;
15: Include a summary of the potential trends in the weather data;
16: Include a summary of the data sources used in the report;
17: Analyze data to determine any upcoming weather changes;
18: Research current weather forecasts for San Francisco;
19: Create a visual representation of the weather report;
20: Publish the report online;
21: Monitor the report for accuracy

*****NEXT TASK*****

3: Format the report for readability;


> Entering new AgentExecutor chain...
Thought: I need to make sure the report is easy to read;
Action: TODO
Action Input: Make the report easy to read

1. Break up the report into sections with clear headings
2. Use bullet points and numbered lists to organize information
3. Use short, concise sentences
4. Use simple language and avoid jargon
5. Include visuals such as charts, graphs, and diagrams to illustrate points
6. Use bold and italicized text to emphasize key points
7. Include a table of contents and page numbers
8. Use a consistent font and font size throughout the report
9. Include a summary at the end of the report
10. Proofread the report for typos and errors I now know the final answer
Final Answer: The report should be formatted for readability by breaking it up into sections with clear headings, using bullet points and numbered lists to organize information, using short, concise sentences, using simple language and avoiding jargon, including visuals such as charts, graphs, and diagrams to illustrate points, using bold and italicized text to emphasize key points, including a table of contents and page numbers, using a consistent font and font size throughout the report, including a summary at the end of the report, and proofreading the report for typos and errors.

> Finished chain.

*****TASK RESULT*****

The report should be formatted for readability by breaking it up into sections with clear headings, using bullet points and numbered lists to organize information, using short, concise sentences, using simple language and avoiding jargon, including visuals such as charts, graphs, and diagrams to illustrate points, using bold and italicized text to emphasize key points, including a table of contents and page numbers, using a consistent font and font size throughout the report, including a summary at the end of the report, and proofreading the report for typos and errors.

*****TASK ENDING*****

```

```pycon
{'objective': 'Write a weather report for SF today'}
```