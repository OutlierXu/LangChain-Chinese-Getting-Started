# AzureOpenAI

让我们加载OpenAI Embedding类，并将环境变量设置为指示使用Azure端点。

```python
# set the environment variables needed for openai package to know to reach out to azure
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
```

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(deployment="your-embeddings-deployment-name")
```

```python
text = "This is a test document."
```

```python
query_result = embeddings.embed_query(text)
```

```python
doc_result = embeddings.embed_documents([text])
```