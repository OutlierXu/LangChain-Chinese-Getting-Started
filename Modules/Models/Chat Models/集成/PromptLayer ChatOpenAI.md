# PromptLayer ChatOpenAI
本示例展示如何连接到PromptLayer以开始记录ChatOpenAI请求。

## 安装PromptLayer

`promptlayer`软件包是将PromptLayer与OpenAI一起使用所必需的。使用pip安装`promptlayer`。

```bash
pip install promptlayer
```

## Imports
```python
import os
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.schema import HumanMessage
```
## 设置环境API密钥
您可以在www.example.com上通过单击导航栏中的设置齿轮来创建PromptLayer API密钥。

将其设置为名为的环境变量`PROMPTLAYER_API_KEY`。
```python
os.environ["PROMPTLAYER_API_KEY"] = "**********"
```

## 像平常一样使用PromptLayerOpenAI LLM
您可以选择传入`pl_tags`以使用PromptLayer的标记功能跟踪您的请求。
```python
chat = PromptLayerChatOpenAI(pl_tags=["langchain"])
chat([HumanMessage(content="I am a cat and I want")])
```

```python
AIMessage(content='to take a nap in a cozy spot. I search around for a suitable place and finally settle on a soft cushion on the window sill. I curl up into a ball and close my eyes, relishing the warmth of the sun on my fur. As I drift off to sleep, I can hear the birds chirping outside and feel the gentle breeze blowing through the window. This is the life of a contented cat.', additional_kwargs={})
```
上面的请求现在应该出现在你的[PromptLayer仪表板上](https://promptlayer.com/)。

##  使用PromptLayer轨迹
如果您希望使用任何PromptLayer跟踪功能，则需要在实例化PromptLayer LLM时传递参数`return_pl_id`以获取请求ID。
```python
chat = PromptLayerChatOpenAI(return_pl_id=True)
chat_results = chat.generate([[HumanMessage(content="I am a cat and I want")]])

for res in chat_results.generations:
    pl_request_id = res[0].generation_info["pl_request_id"]
    promptlayer.track.score(request_id=pl_request_id, score=100)
```

使用此选项可以在PromptLayer仪表板中跟踪模型的性能。

如果使用的是提示模板，则还可以将模板附加到请求。总的来说，这使您有机会在PromptLayer仪表板中跟踪不同模板和模型的性能。