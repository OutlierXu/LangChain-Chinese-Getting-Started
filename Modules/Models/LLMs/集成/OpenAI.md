# OpenAI
OpenAI提供了一系列具有不同级别功率的模型，适用于不同的任务。

这个例子讲述了如何使用LangChain与`OpenAI`模型交互.
```python
# get a token: https://platform.openai.com/account/api-keys

from getpass import getpass

OPENAI_API_KEY = getpass()
```
```pycon
 ········
```
```python
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)
```

```pycon
' Justin Bieber was born in 1994, so we are looking for the Super Bowl winner from that year. The Super Bowl in 1994 was Super Bowl XXVIII, and the winner was the Dallas Cowboys.'
```

如果位于显式代理之后，则可以使用OPENAI_PROXY环境变量来传递.

```python
os.environ[“OPENAI_PROXY”] = “http://proxy.yourcompany.com:8080”
os.environ[“OPENAI_PROXY”] =“http://proxy.yourcompany.com:8080“
```