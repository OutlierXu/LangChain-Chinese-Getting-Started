# 如何编写自定义LLM包装器
这个笔记介绍了如何创建一个自定义的LLM包装器，如果你想使用自己的LLM或一个不同于LangChain支持的包装器。

自定义LLM只需要实现一件事：

1. 一个`_call`方法，它接受一个字符串，一些可选的停止词，并返回一个字符串.

它还可以实现第二个可选功能：

一个`_identifying_params`属性，用于帮助打印此类的属性。应该返回一个字典。

让我们实现一个非常简单的自定义LLM，它只返回输入的前N个字符。

```python
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class CustomLLM(LLM):
    
    n: int
        
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.n]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
```

我们现在可以像任何其他LLM一样使用它。
```python
llm = CustomLLM(n=10)
```
```pycon
llm("This is a foobar thing")
```
```pycon
'This is a '
```

我们还可以打印LLM并查看其自定义打印。

```python
print(llm)
```
```pycon
CustomLLM
Params: {'n': 10}
```
