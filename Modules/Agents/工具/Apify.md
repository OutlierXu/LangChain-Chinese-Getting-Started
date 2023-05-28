# Apify

这个笔记展示了如何使用LangChain的Apify集成。

[Apify](https://apify.com/)是一个用于Web抓取和数据提取的云平台，它提供了一个由一千多个称为Actors的现成应用程序组成的[生态系统](https://apify.com/store)，用于各种Web抓取，爬虫和数据提取实践。例如，您可以使用它来提取Google搜索结果，Instagram和Facebook个人资料，亚马逊或Shopify的产品，Google地图评论等等的。

在本例中，我们将使用Website [Content Crawler Actor](https://apify.com/apify/website-content-crawler)，它可以深度抓取文档、知识库、帮助中心或博客等网站，并从网页中提取文本内容。然后，我们将文档输入到向量索引中，并回答问题。

```bash
#!pip install apify-client
```

首先，将 `ApifyWrapper` 导入你的代码：

```python
from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
```
使用您的[Apify API](https://console.apify.com/account/integrations)令牌初始化它，并且出于本示例的目的，还使用您的OpenAI API密钥：

```python
import os
os.environ["OPENAI_API_KEY"] = "Your OpenAI API key"
os.environ["APIFY_API_TOKEN"] = "Your Apify API token"

apify = ApifyWrapper()
```

然后运行Actor，等待它完成，并将其结果从Apify数据集提取到LangChain文档加载器中。

请注意，如果Apify数据集中已经有一些结果，则可以使用 ApifyDatasetLoader 直接加载它们，如[本笔记所示](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/apify_dataset.html)。在该笔记本中，您还可以找到 `dataset_mapping_function` 的解释，它用于将Apify数据集记录中的字段映射到LangChain `Document` 字段。
```python
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://python.langchain.com/en/latest/"}]},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)
```

从抓取的文档初始化向量索引：
```python
index = VectorstoreIndexCreator().from_loaders([loader])
```
最后，查询vector index：

```python
query = "What is LangChain?"
result = index.query_with_sources(query)
```

```python
print(result["answer"])
print(result["sources"])
```

```python
 LangChain is a standard interface through which you can interact with a variety of large language models (LLMs). It provides modules that can be used to build language model applications, and it also provides chains and agents with memory capabilities.

https://python.langchain.com/en/latest/modules/models/llms.html, https://python.langchain.com/en/latest/getting_started/getting_started.html
```