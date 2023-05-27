# 如何（以及为什么）使用人工输入LLM

与伪LLM类似，LangChain提供了一个伪LLM类，可用于测试，调试或教育目的。这允许您模拟对LLM的调用，并模拟人类在收到prompts时将如何响应。   

在这个笔记中，我们将学习如何使用它。

我们首先在代理中使用HumanInputLLM。

```python
from langchain.llms.human import HumanInputLLM

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
```
由于我们将在本笔记中使用`WikipediaQueryRun`工具，因此如果您还没有安`wikipedia`该软件包.

```bash
pip install wikipedia
```
```python
tools = load_tools(["wikipedia"])
llm = HumanInputLLM(prompt_func=lambda prompt: print(f"\n===PROMPT====\n{prompt}\n=====END OF PROMPT======"))

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is 'Bocchi the Rock!'?")
```

```pycon
> Entering new AgentExecutor chain...

===PROMPT====
Answer the following questions as best you can. You have access to the following tools:

Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Wikipedia]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: What is 'Bocchi the Rock!'?
Thought:
=====END OF PROMPT======
I need to use a tool.
Action: Wikipedia
Action Input: Bocchi the Rock!, Japanese four-panel manga and anime series.
Observation: Page: Bocchi the Rock!
Summary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Bocchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.
An anime television series adaptation produced by CloverWorks aired from October to December 2022. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.

Page: Manga Time Kirara
Summary: Manga Time Kirara (まんがタイムきらら, Manga Taimu Kirara) is a Japanese seinen manga magazine published by Houbunsha which mainly serializes four-panel manga. The magazine is sold on the ninth of each month and was first published as a special edition of Manga Time, another Houbunsha magazine, on May 17, 2002. Characters from this magazine have appeared in a crossover role-playing game called Kirara Fantasia.

Page: Manga Time Kirara Max
Summary: Manga Time Kirara Max (まんがタイムきららMAX) is a Japanese four-panel seinen manga magazine published by Houbunsha. It is the third magazine of the "Kirara" series, after "Manga Time Kirara" and "Manga Time Kirara Carat". The first issue was released on September 29, 2004. Currently the magazine is released on the 19th of each month.
Thought:
===PROMPT====
Answer the following questions as best you can. You have access to the following tools:

Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Wikipedia]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: What is 'Bocchi the Rock!'?
Thought:I need to use a tool.
Action: Wikipedia
Action Input: Bocchi the Rock!, Japanese four-panel manga and anime series.
Observation: Page: Bocchi the Rock!
Summary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Bocchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.
An anime television series adaptation produced by CloverWorks aired from October to December 2022. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.

Page: Manga Time Kirara
Summary: Manga Time Kirara (まんがタイムきらら, Manga Taimu Kirara) is a Japanese seinen manga magazine published by Houbunsha which mainly serializes four-panel manga. The magazine is sold on the ninth of each month and was first published as a special edition of Manga Time, another Houbunsha magazine, on May 17, 2002. Characters from this magazine have appeared in a crossover role-playing game called Kirara Fantasia.

Page: Manga Time Kirara Max
Summary: Manga Time Kirara Max (まんがタイムきららMAX) is a Japanese four-panel seinen manga magazine published by Houbunsha. It is the third magazine of the "Kirara" series, after "Manga Time Kirara" and "Manga Time Kirara Carat". The first issue was released on September 29, 2004. Currently the magazine is released on the 19th of each month.
Thought:
=====END OF PROMPT======
These are not relevant articles.
Action: Wikipedia
Action Input: Bocchi the Rock!, Japanese four-panel manga series written and illustrated by Aki Hamaji.
Observation: Page: Bocchi the Rock!
Summary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Bocchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.
An anime television series adaptation produced by CloverWorks aired from October to December 2022. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.
Thought:
===PROMPT====
Answer the following questions as best you can. You have access to the following tools:

Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Wikipedia]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: What is 'Bocchi the Rock!'?
Thought:I need to use a tool.
Action: Wikipedia
Action Input: Bocchi the Rock!, Japanese four-panel manga and anime series.
Observation: Page: Bocchi the Rock!
Summary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Bocchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.
An anime television series adaptation produced by CloverWorks aired from October to December 2022. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.

Page: Manga Time Kirara
Summary: Manga Time Kirara (まんがタイムきらら, Manga Taimu Kirara) is a Japanese seinen manga magazine published by Houbunsha which mainly serializes four-panel manga. The magazine is sold on the ninth of each month and was first published as a special edition of Manga Time, another Houbunsha magazine, on May 17, 2002. Characters from this magazine have appeared in a crossover role-playing game called Kirara Fantasia.

Page: Manga Time Kirara Max
Summary: Manga Time Kirara Max (まんがタイムきららMAX) is a Japanese four-panel seinen manga magazine published by Houbunsha. It is the third magazine of the "Kirara" series, after "Manga Time Kirara" and "Manga Time Kirara Carat". The first issue was released on September 29, 2004. Currently the magazine is released on the 19th of each month.
Thought:These are not relevant articles.
Action: Wikipedia
Action Input: Bocchi the Rock!, Japanese four-panel manga series written and illustrated by Aki Hamaji.
Observation: Page: Bocchi the Rock!
Summary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Bocchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.
An anime television series adaptation produced by CloverWorks aired from October to December 2022. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.
Thought:
=====END OF PROMPT======
It worked.
Final Answer: Bocchi the Rock! is a four-panel manga series and anime television series. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.

> Finished chain.
```

```pycon
"Bocchi the Rock! is a four-panel manga series and anime television series. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim."
```