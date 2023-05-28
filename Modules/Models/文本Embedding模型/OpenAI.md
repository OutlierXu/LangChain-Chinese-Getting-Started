# OpenAI

让我们加载OpenAI Embedding类。


```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
```
让我们用第一代模型加载OpenAI Embedding类（例如text-search-ada-doc-001/text-search-ada-query-001）。注意：这些不是推荐型号-请参阅[此处](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)

```python
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


text = "This is a test document."
query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text])


# if you are behind an explicit proxy, you can use the OPENAI_PROXY environment variable to pass through
os.environ["OPENAI_PROXY"] = "http://proxy.yourcompany.com:8080"
```

