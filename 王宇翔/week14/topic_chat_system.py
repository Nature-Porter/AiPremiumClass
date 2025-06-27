from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class TopicChatSystem:
    """特定主题聊天系统，支持多轮对话"""
    
    def __init__(self):
        """初始化聊天系统"""
        load_dotenv(find_dotenv())
        
        # 初始化LLM模型
        self.model = ChatOpenAI(
            model="glm-4-flash-250414",
            base_url=os.environ['base_url'],
            api_key=os.environ['api_key'],
            temperature=0.7
        )
        
        # 定义支持的主题
        self.topics = {
            "健康": "你是一个专业的健康咨询顾问，能够提供科学、准确的健康建议和知识。请用专业但易懂的语言回答用户的健康相关问题。",
            "科技": "你是一个科技领域的专家，对最新的科技发展、编程技术、人工智能等有深入了解。请用通俗易懂的方式解释复杂的科技概念。",
            "教育": "你是一个经验丰富的教育专家，能够提供学习方法、教育建议和知识解答。请耐心细致地帮助用户解决学习问题。",
            "旅游": "你是一个资深的旅游顾问，对世界各地的旅游景点、文化、美食都有深入了解。请为用户提供实用的旅游建议和信息。",
            "美食": "你是一个美食专家，精通各种菜系和烹饪技巧。请为用户介绍美食文化、制作方法和饮食建议。"
        }
        
        # 存储不同会话的聊天历史
        self.store = {}
        
        # 当前主题
        self.current_topic = None
        
        # 解析器
        self.parser = StrOutputParser()
        
        # 构建chain
        self._build_chain()
    
    def _build_chain(self):
        """构建langchain处理链"""
        # 创建带有占位符的prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{topic_instruction}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # 构建chain
        self.chain = self.prompt | self.model | self.parser
        
        # 注入聊天历史功能
        self.with_msg_hist = RunnableWithMessageHistory(
            self.chain,
            get_session_history=self._get_session_history,
            input_messages_key="messages"
        )
    
    def _get_session_history(self, session_id):
        """根据session_id获取聊天历史"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def set_topic(self, topic_name):
        """设置聊天主题"""
        if topic_name in self.topics:
            self.current_topic = topic_name
            print(f"✅ 已切换到 '{topic_name}' 主题聊天模式")
            print(f"📝 {self.topics[topic_name]}")
            return True
        else:
            print(f"❌ 不支持的主题：{topic_name}")
            print(f"🔍 支持的主题：{', '.join(self.topics.keys())}")
            return False
    
    def get_available_topics(self):
        """获取所有可用主题"""
        return list(self.topics.keys())
    
    def chat(self, user_input, session_id="default"):
        """进行对话"""
        if not self.current_topic:
            return "❌ 请先选择聊天主题！使用 set_topic() 方法设置主题。"
        
        try:
            # 获取当前主题的系统指令
            topic_instruction = self.topics[self.current_topic]
            
            # 调用链进行对话
            response = self.with_msg_hist.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "topic_instruction": topic_instruction
                },
                config={'configurable': {'session_id': session_id}}
            )
            
            return response
            
        except Exception as e:
            return f"❌ 对话过程中出现错误：{str(e)}"
    
    def clear_history(self, session_id="default"):
        """清除指定会话的聊天历史"""
        if session_id in self.store:
            del self.store[session_id]
            print(f"✅ 已清除会话 {session_id} 的聊天历史")
        else:
            print(f"⚠️ 会话 {session_id} 不存在")
    
    def show_history(self, session_id="default"):
        """显示聊天历史"""
        if session_id in self.store:
            history = self.store[session_id]
            print(f"\n=== 会话 {session_id} 的聊天历史 ===")
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    print(f"👤 用户: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"🤖 AI: {msg.content}")
            print("=" * 30)
        else:
            print(f"⚠️ 会话 {session_id} 没有聊天历史")


def main():
    """主函数 - 交互式聊天界面"""
    print("🎉 欢迎使用特定主题聊天系统！")
    print("=" * 50)
    
    # 创建聊天系统实例
    chat_system = TopicChatSystem()
    
    # 显示可用主题
    topics = chat_system.get_available_topics()
    print("📋 可用主题:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")
    
    print("\n💡 命令说明:")
    print("  - 输入数字选择主题")
    print("  - 输入 'topics' 查看所有主题")
    print("  - 输入 'history' 查看聊天历史")
    print("  - 输入 'clear' 清除聊天历史")
    print("  - 输入 'quit' 退出系统")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 请输入: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 感谢使用，再见！")
            break
        
        elif user_input.lower() == 'topics':
            print("\n📋 可用主题:")
            for i, topic in enumerate(topics, 1):
                print(f"  {i}. {topic}")
        
        elif user_input.lower() == 'history':
            chat_system.show_history()
        
        elif user_input.lower() == 'clear':
            chat_system.clear_history()
        
        elif user_input.isdigit():
            topic_index = int(user_input) - 1
            if 0 <= topic_index < len(topics):
                selected_topic = topics[topic_index]
                chat_system.set_topic(selected_topic)
            else:
                print("❌ 无效的主题编号")
        
        elif user_input:
            # 进行对话
            response = chat_system.chat(user_input)
            print(f"\n🤖 AI回复: {response}")
        
        else:
            print("⚠️ 请输入有效内容")


if __name__ == '__main__':
    main() 