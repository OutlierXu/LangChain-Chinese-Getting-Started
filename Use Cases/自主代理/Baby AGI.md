# BabyAGI 用户指南

此笔记本演示了 Yohei Nakajima 如何实施 BabyAGI。 BabyAGI 是一种人工智能代理，可以根据给定目标生成并假装执行任务。

本指南将帮助您了解并创建自己的递归代理的组件。

尽管 BabyAGI 使用特定的向量存储/模型提供者（Pinecone、OpenAI），但使用 LangChain 实现它的好处之一是您可以轻松地将它们换成不同的选项。在此实现中，我们使用 FAISS vectorstore（因为它在本地运行且免费）。


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

根据您使用的向量存储，此步骤可能看起来有所不同。
```python
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
```

## 运行 BabyAGI

现在是创建 BabyAGI 控制器并观察它尝试实现您的目标的时候了。


```python
OBJECTIVE = "Write a weather report for SF today"

llm = OpenAI(temperature=0)

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)


baby_agi({"objective": OBJECTIVE})
```
```pycon

*****TASK LIST*****

1: Make a todo list

*****NEXT TASK*****

1: Make a todo list

*****TASK RESULT*****



1. Check the weather forecast for San Francisco today
2. Make note of the temperature, humidity, wind speed, and other relevant weather conditions
3. Write a weather report summarizing the forecast
4. Check for any weather alerts or warnings
5. Share the report with the relevant stakeholders

*****TASK LIST*****

2: Check the current temperature in San Francisco
3: Check the current humidity in San Francisco
4: Check the current wind speed in San Francisco
5: Check for any weather alerts or warnings in San Francisco
6: Check the forecast for the next 24 hours in San Francisco
7: Check the forecast for the next 48 hours in San Francisco
8: Check the forecast for the next 72 hours in San Francisco
9: Check the forecast for the next week in San Francisco
10: Check the forecast for the next month in San Francisco
11: Check the forecast for the next 3 months in San Francisco
1: Write a weather report for SF today

*****NEXT TASK*****

2: Check the current temperature in San Francisco

*****TASK RESULT*****



I will check the current temperature in San Francisco. I will use an online weather service to get the most up-to-date information.

*****TASK LIST*****

3: Check the current UV index in San Francisco.
4: Check the current air quality in San Francisco.
5: Check the current precipitation levels in San Francisco.
6: Check the current cloud cover in San Francisco.
7: Check the current barometric pressure in San Francisco.
8: Check the current dew point in San Francisco.
9: Check the current wind direction in San Francisco.
10: Check the current humidity levels in San Francisco.
1: Check the current temperature in San Francisco to the average temperature for this time of year.
2: Check the current visibility in San Francisco.
11: Write a weather report for SF today.

*****NEXT TASK*****

3: Check the current UV index in San Francisco.

*****TASK RESULT*****



The current UV index in San Francisco is moderate. The UV index is expected to remain at moderate levels throughout the day. It is recommended to wear sunscreen and protective clothing when outdoors.

*****TASK ENDING*****

```

{'objective': 'Write a weather report for SF today'}
