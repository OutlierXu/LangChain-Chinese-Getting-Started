# 如何序列化LLM类

本笔记本介绍了如何将LLM配置写入磁盘以及从磁盘读取LLM配置。如果要保存给定LLM的配置（例如，提供者、temperature等）时，这个操作会很有用。
```python
from langchain.llms import OpenAI
from langchain.llms.loading import load_llm
```

## Loading 加载
首先，让我们从磁盘加载一个LLM。LLM可以以两种格式保存在磁盘上：json或yaml。无论扩展名如何，它们都以相同的方式加载。
```bash
!cat llm.json
```

```json
{
    "model_name": "text-davinci-003",
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "n": 1,
    "best_of": 1,
    "request_timeout": null,
    "_type": "openai"
}
```
```python
llm = load_llm("llm.json")
```

```bash
!cat llm.yaml
```

```yaml
_type: openai
best_of: 1
frequency_penalty: 0.0
max_tokens: 256
model_name: text-davinci-003
n: 1
presence_penalty: 0.0
request_timeout: null
temperature: 0.7
top_p: 1.0
```
```python
llm = load_llm("llm.yaml")
```

## Saving 保存

如果您想从内存中的LLM转换到它的序列化版本，可以通过调用方法`.save`轻松完成。同样，它同时支持json和yaml。
```python
llm.save("llm.json")

llm.save("llm.yaml")
```