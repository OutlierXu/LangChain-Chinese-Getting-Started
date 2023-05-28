# Agent Types

代理使用LLM来确定要采取哪些操作以及采取的顺序。动作可以是使用工具并观察其输出，也可以是向用户返回响应。以下是LangChain中可用的代理。

## zero-shot-react-description

该代理使用ReAct框架来确定仅基于工具的描述来使用哪个工具。可以提供任何数量的工具。此代理要求为每个工具提供描述。


## react-docstore
该代理使用ReAct框架与Docstore进行交互。必须提供两种工具： `Search` 工具和 `Lookup` 工具（它们的名称必须完全相同）。 `Search` 工具应该搜索文档，而 `Lookup` 工具应该在最近找到的文档中查找术语。这个代理相当于原始的[ReAct论文](https://arxiv.org/pdf/2210.03629.pdf)，特别是维基百科的例子。

## self-ask-with-search
此代理使用命名为 `Intermediate Answer` 的单个工具。这个工具应该能够查找问题的事实答案。这个代理相当于原始的[self ask with search paper](https://ofir.io/self-ask.pdf)，其中提供了Google搜索API作为工具。

## conversational-react-description

此代理设计用于会话设置。设计好的prompt旨在使agent有用的且易于交谈。它使用ReAct框架来决定使用哪个工具，并使用内存来记住以前的对话。