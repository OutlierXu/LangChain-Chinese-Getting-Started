# Sentence Transformers Embeddings 句子转换器嵌入

 [SentenceTransformers](https://www.sbert.net/)embeddings使用 `HuggingFaceEmbeddings` 集成调用。我们还为更熟悉直接使用该软件包的用户添加了 `SentenceTransformerEmbeddings` 的别名。
 
SentenceTransformers是一个可以生成文本和图像嵌入的python包，起源于 [Sentence-BERT](https://arxiv.org/abs/1908.10084)

```bash
!pip install sentence_transformers > /dev/null
```

```pycon
[notice] A new release of pip is available: 23.0.1 -> 23.1.1
[notice] To update, run: pip install --upgrade pip
```

```python
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 
```

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

```python
text = "This is a test document."
```

```python
query_result = embeddings.embed_query(text)
```

```python
doc_result = embeddings.embed_documents([text, "This is not a test document."])
```