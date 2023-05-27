# Anthropic

本手册介绍了如何开始使用Anthropic聊天模型。

```python
from langchain.chat_models import ChatAnthropic
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
chat = ChatAnthropic()
```
```python
messages = [
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]
chat(messages)
```

```python
AIMessage(content=" J'aime programmer. ", additional_kwargs={})
```

## `ChatAnthropic`还支持异步和流式传输功能：
```python
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
```

```python
await chat.agenerate([messages])
```

```python
LLMResult(generations=[[ChatGeneration(text=" J'aime la programmation.", generation_info=None, message=AIMessage(content=" J'aime la programmation.", additional_kwargs={}))]], llm_output={})
```

```python
chat = ChatAnthropic(streaming=True, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
chat(messages)
```

```python
 J'adore programmer.
```

```python
AIMessage(content=" J'adore programmer.", additional_kwargs={})
```