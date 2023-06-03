# Agent Simulations 代理模拟

代理模拟涉及多个代理之间相互交互。代理模拟通常涉及两个主要部分：
- Long Term Memory 长期记忆
- Simulation Environment 仿真环境


代理模拟（或部分代理模拟）的具体实现包括：

## 一个代理的模拟
- [模拟环境：Gymnasium](../代理模拟/模拟环境:Gymnasium.md)：一个示例，说明如何使用 [Gymnasium](https://gymnasium.farama.org/)（以前称为 [OpenAI Gym](https://github.com/openai/gym)）创建简单的代理-环境交互循环。

## 两个代理的模拟
- [CAMEL](../代理模拟/CAMEL 角色扮演自治合作代理.md)：CAMEL（Communicative Agents for “Mind” Exploration of Large Scale Language Model Society）论文的实现，其中两个代理相互通信。
- [两人 D&D](../代理模拟/双人龙与地下城.md)：一个示例，说明如何使用两个代理的通用模拟器来实现流行的龙与地下城角色扮演游戏的变体。
- [Agent Debates with Tools](../代理模拟/Agent Debates with Tools.md)：一个示例，说明如何使对话代理能够使用工具来通知他们的响应。
## 多个代理的模拟



- [多人 D&D](https://python.langchain.com/en/latest/use_cases/agent_simulations/multi_player_dnd.html)：一个示例，说明如何将通用对话模拟器用于具有自定义说话者顺序的多个对话代理，以流行的龙与地下城角色扮演游戏的变体进行说明。
- [分散式说话人选择](https://python.langchain.com/en/latest/use_cases/agent_simulations/multiagent_bidding.html)：如何在没有固定的发言时间安排的情况下实施多代理对话的示例。取而代之的是，代理人通过输出出价来自己决定谁发言。这个例子展示了如何在一场虚构的总统辩论中做到这一点。
- [威权发言人选择](https://python.langchain.com/en/latest/use_cases/agent_simulations/multiagent_authoritarian.html)：如何实施多代理对话的示例，其中特权代理指示谁说什么。此示例还展示了如何使特权代理能够确定对话何时终止。此示例说明如何在虚构新闻节目的上下文中执行此操作。
- [模拟环境：PettingZoo](https://python.langchain.com/en/latest/use_cases/agent_simulations/petting_zoo.html)：如何使用 PettingZoo（Gymnasium 的多代理版本）为多个代理创建代理-环境交互循环的示例。
- [生成代理](https://python.langchain.com/en/latest/use_cases/agent_simulations/characters.html)：本笔记实现一个生成代理基于论文[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)：Park 等人的人类行为交互模拟。阿尔。