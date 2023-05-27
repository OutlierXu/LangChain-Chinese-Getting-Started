# Hugging Face Hub

[Hugging Face Hub](https://huggingface.co/docs/hub/index)是一个平台，拥有超过12万个模型，2万个数据集和5万个演示应用程序（Spaces），所有这些都是开源和公开的，在一个在线平台上，人们可以轻松地协作并共同构建ML。

此示例演示如何连接到Hugging Face Hub.
要使用`Hugging Face Hub`，您应该安装python包。
```bash
!pip install huggingface_hub > /dev/null
```

```python
# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token

from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass()
```

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
```

#### Select a Model 选择型号
```python
from langchain import HuggingFaceHub

repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
```

```python
from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))
```

## 示例
下面是一些可以通过Hugging Face Hub集成访问的模型示例。

#### StableLM, by Stability AI StableLM，通过稳定性AI
请参阅[Stability AI](https://huggingface.co/stabilityai)的组织页面以获取可用模型的列表。
```python
repo_id = "stabilityai/stablelm-tuned-alpha-3b"
# Others include stabilityai/stablelm-base-alpha-3b
# as well as 7B parameter versions
```
```python
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
```
```python
# Reuse the prompt and question from above.
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))
```
#### Dolly, by DataBricks
有关可用模型的列表，请参见[DataBricks](https://huggingface.co/databricks)组织页面。
```python
from langchain import HuggingFaceHub

repo_id = "databricks/dolly-v2-3b"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
```
```python
# Reuse the prompt and question from above.
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))
```

#### Camel, by Writer
有关可用模型的列表，请参见[Writer](https://huggingface.co/Writer)组织页面。
```python
from langchain import HuggingFaceHub

repo_id = "Writer/camel-5b-hf" # See https://huggingface.co/Writer for other options
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
```

```python
# Reuse the prompt and question from above.
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))
```
And many more! 还有更多！