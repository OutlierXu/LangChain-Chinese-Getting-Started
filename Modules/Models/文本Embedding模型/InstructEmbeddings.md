# InstructEmbeddings

让我们加载HuggingFace指令Embeddings类。

```python
from langchain.embeddings import HuggingFaceInstructEmbeddings
```

```python
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)
```

```pycon
load INSTRUCTOR_Transformer
max_seq_length  512
```

```python
text = "This is a test document."
```

```python
query_result = embeddings.embed_query(text)
```