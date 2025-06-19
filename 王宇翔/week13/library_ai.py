# librarian_ai.py
import os
import json
from openai import OpenAI
from mock_lms import MockLMS
from dotenv import load_dotenv, find_dotenv
class LibrarianAI:
    def __init__(self, user_id):
        # 确保API密钥已设置
        load_dotenv(find_dotenv())
    
    # 创建调用客户端
        self.client = OpenAI(
        api_key=os.environ["api_key"],
        base_url=os.environ["base_url"]
        )
        self.lms = MockLMS()
        self.user_id = user_id
        self.system_prompt = """
# 角色与身份 (Role & Persona)
你是一个名为“智图AI”的智能图书管理员。你的性格友好、博学、耐心且高效。你的目标是帮助读者轻松地借阅、归还图书，并根据他们的兴趣发现好书。你的语言风格既要专业，又要充满人文关怀。

# 核心能力 (Core Capabilities)
1.  **图书借还办理 (Book Transaction Processing)**: 你可以处理图书的借阅和归还请求。
2.  **个性化推荐 (Personalized Recommendation)**: 你能根据读者的阅读历史、评分、明确偏好和当前对话内容，为他们推荐图书。
3.  **图书查询 (Book Search)**: 你可以根据书名、作者、ISBN或关键词查询图书信息及其馆藏状态。
4.  **闲聊 (Chat)**: 对于非功能性请求，进行友好对话。

# 约束与输出格式 (Constraints & Output Format)
- 你不能凭空捏造图书信息或读者数据。你所有的知识都基于我提供给你的[图书数据库信息]和[读者信息]。
- **至关重要**: 你必须根据用户意图，生成一个包含`action`的JSON对象。
- **JSON输出结构**:
{
  "action": "ACTION_TYPE",
  "parameters": {
    // 根据action类型填充
  },
  "response_to_user": "给用户的自然语言回复"
}
- **Action类型**: "borrow", "return", "recommend", "search", "chat"
"""

    def _get_context_data(self):
        """获取当前用户和图书的上下文信息"""
        user_info = self.lms.get_user_info(self.user_id)
        all_books = self.lms.get_all_books_for_recommendation()
        
        # 将读者借阅的book_id转换为书名，让LLM更容易理解
        borrowed_book_titles = [self.lms.get_book_info(bid)['title'] for bid in user_info.get('borrowed_books', [])]
        user_info_for_prompt = user_info.copy()
        user_info_for_prompt['borrowed_books'] = borrowed_book_titles

        return json.dumps(user_info_for_prompt, ensure_ascii=False), json.dumps(all_books, ensure_ascii=False)

    def _execute_action(self, llm_response):
        """解析并执行LLM返回的指令"""
        try:
            action_data = json.loads(llm_response)
            action = action_data.get("action")
            params = action_data.get("parameters", {})
            user_response = action_data.get("response_to_user", "我好像有点没理解，能再说一遍吗？")

            if action == "borrow":
                book_id = params.get("book_id")
                if not book_id: return "您想借哪本书呢？我需要书的ID，比如'B001'。"
                success, message = self.lms.borrow_book(self.user_id, book_id)
                return f"{user_response}\n[系统消息: {message}]"
            
            elif action == "return":
                book_id = params.get("book_id")
                if not book_id: return "您想还哪本书呢？我需要书的ID。"
                success, message = self.lms.return_book(self.user_id, book_id)
                return f"{user_response}\n[系统消息: {message}]"
            
            # 对于search, recommend, chat，我们直接返回LLM生成的回复
            # 因为这些操作不改变后端状态，或者其状态已经在Prompt中提供给了LLM
            else:
                return user_response

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[错误] LLM返回的JSON格式不正确或缺少字段: {llm_response}")
            return "抱歉，我的内部系统出了一点小问题，请稍后再试。"


    def chat(self, user_input):
        """主交互函数"""
        user_info_json, book_data_json = self._get_context_data()
        
        prompt = f"""
# 上下文信息
[当前读者信息 user_id={self.user_id}]:
{user_info_json}

[图书馆藏信息]:
{book_data_json}

# 用户问题
{user_input}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash-250414",  # 推荐使用支持JSON模式的较新模型
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"} # 开启JSON模式
            )
            llm_output = response.choices[0].message.content
            print(f"--- LLM原始输出 ---\n{llm_output}\n--------------------") # 调试用
            
            final_response = self._execute_action(llm_output)
            return final_response

        except Exception as e:
            print(f"[错误] 调用OpenAI API失败: {e}")
            return "抱歉，我暂时无法连接到服务，请稍后再试。"