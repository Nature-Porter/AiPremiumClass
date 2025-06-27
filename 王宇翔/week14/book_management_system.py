from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
import json


class BookManagementSystem:
    """图书管理系统 - 基于图书简介提供咨询服务"""
    
    def __init__(self):
        """初始化图书管理系统"""
        load_dotenv(find_dotenv())
        
        # 初始化LLM模型
        self.model = ChatOpenAI(
            model="glm-4-flash-250414",
            base_url=os.environ['base_url'],
            api_key=os.environ['api_key'],
            temperature=0.7
        )
        
        # 图书数据库（模拟）
        self.books_db = {
            "1": {
                "title": "活着",
                "author": "余华",
                "category": "文学小说",
                "isbn": "9787506365437",
                "status": "可借阅",
                "description": "《活着》讲述了一个人和他的命运之间的友情，这是最为感人的友情，因为他们互相感激，同时也互相仇恨；他们谁也无法抛弃对方，同时谁也没有理由抱怨对方。小说记录了一个人的一生，一个历史时期的社会变迁。",
                "tags": ["现实主义", "人生哲理", "中国文学", "苦难与坚韧"],
                "suitable_readers": "适合对人生有思考的读者，文学爱好者",
                "reading_difficulty": "中等"
            },
            "2": {
                "title": "百年孤独",
                "author": "加西亚·马尔克斯",
                "category": "文学小说",
                "isbn": "9787544291170",
                "status": "可借阅",
                "description": "《百年孤独》是魔幻现实主义文学的代表作，描写了布恩迪亚家族七代人的传奇故事，以及加勒比海沿岸小镇马孔多的百年兴衰，反映了拉丁美洲一个世纪以来风云变幻的历史。",
                "tags": ["魔幻现实主义", "拉美文学", "家族史", "孤独主题"],
                "suitable_readers": "文学爱好者，对拉美文化感兴趣的读者",
                "reading_difficulty": "较高"
            },
            "3": {
                "title": "Python编程：从入门到实践",
                "author": "埃里克·马瑟斯",
                "category": "计算机技术",
                "isbn": "9787115428028",
                "status": "可借阅",
                "description": "本书是一本针对所有层次Python读者而作的Python入门书。全书分两部分：第一部分介绍用Python编程所必须了解的基本概念，第二部分将理论付诸实践，讲解如何开发三个项目。",
                "tags": ["编程", "Python", "入门教程", "实践项目"],
                "suitable_readers": "编程初学者，想学习Python的读者",
                "reading_difficulty": "中等"
            },
            "4": {
                "title": "人工智能：一种现代的方法",
                "author": "斯图尔特·罗素",
                "category": "计算机技术",
                "isbn": "9787111543367",
                "status": "已借出",
                "description": "这是一本全面而系统的人工智能教材，涵盖了人工智能的理论基础、算法实现和应用实例。从智能代理开始，全面介绍了人工智能的核心主题。",
                "tags": ["人工智能", "机器学习", "算法", "理论基础"],
                "suitable_readers": "计算机专业学生，AI研究者",
                "reading_difficulty": "高"
            },
            "5": {
                "title": "史记",
                "author": "司马迁",
                "category": "历史",
                "isbn": "9787101003048",
                "status": "可借阅",
                "description": "《史记》是西汉史学家司马迁撰写的纪传体史书，是中国历史上第一部纪传体通史，记载了从上古传说中的黄帝时代到汉武帝太初四年共3000多年的历史。",
                "tags": ["中国古代史", "纪传体", "经典史书", "文史哲"],
                "suitable_readers": "历史爱好者，古典文学研究者",
                "reading_difficulty": "较高"
            }
        }
        
        # 存储不同用户的聊天历史
        self.user_sessions = {}
        
        # 解析器
        self.parser = StrOutputParser()
        
        # 构建咨询chain
        self._build_consultation_chain()
    
    def _build_consultation_chain(self):
        """构建图书咨询链"""
        # 创建图书咨询的prompt template
        self.consultation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的图书管理员和阅读顾问。你的任务是：
            1. 根据用户的需求推荐合适的图书
            2. 为用户详细介绍图书内容和特点
            3. 提供阅读建议和指导
            4. 回答关于图书的各种问题
            
            请用友好、专业的语气回答用户问题。在推荐图书时，要考虑读者的兴趣、阅读水平和需求。
            
            以下是当前图书库存信息：
            {books_info}"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # 构建chain
        self.consultation_chain = self.consultation_prompt | self.model | self.parser
        
        # 注入聊天历史功能
        self.with_msg_hist = RunnableWithMessageHistory(
            self.consultation_chain,
            get_session_history=self._get_session_history,
            input_messages_key="messages"
        )
    
    def _get_session_history(self, user_id):
        """根据用户ID获取聊天历史"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = InMemoryChatMessageHistory()
        return self.user_sessions[user_id]
    
    def _format_books_info(self):
        """格式化图书信息为字符串"""
        books_info = "图书库存信息：\n\n"
        for book_id, book in self.books_db.items():
            books_info += f"书籍ID: {book_id}\n"
            books_info += f"书名: {book['title']}\n"
            books_info += f"作者: {book['author']}\n"
            books_info += f"分类: {book['category']}\n"
            books_info += f"借阅状态: {book['status']}\n"
            books_info += f"简介: {book['description']}\n"
            books_info += f"标签: {', '.join(book['tags'])}\n"
            books_info += f"适合读者: {book['suitable_readers']}\n"
            books_info += f"阅读难度: {book['reading_difficulty']}\n"
            books_info += "-" * 50 + "\n"
        
        return books_info
    
    def search_books(self, keyword=None, category=None, status=None):
        """搜索图书"""
        results = []
        
        for book_id, book in self.books_db.items():
            match = True
            
            # 关键词搜索
            if keyword:
                keyword_lower = keyword.lower()
                if not (keyword_lower in book['title'].lower() or 
                       keyword_lower in book['author'].lower() or
                       keyword_lower in book['description'].lower() or
                       any(keyword_lower in tag.lower() for tag in book['tags'])):
                    match = False
            
            # 分类筛选
            if category and book['category'] != category:
                match = False
            
            # 状态筛选
            if status and book['status'] != status:
                match = False
            
            if match:
                results.append((book_id, book))
        
        return results
    
    def get_book_details(self, book_id):
        """获取图书详细信息"""
        if book_id in self.books_db:
            return self.books_db[book_id]
        return None
    
    def borrow_book(self, book_id, user_name):
        """借阅图书"""
        if book_id not in self.books_db:
            return False, "图书不存在"
        
        book = self.books_db[book_id]
        if book['status'] != "可借阅":
            return False, f"图书《{book['title']}》当前不可借阅，状态：{book['status']}"
        
        # 更新借阅状态
        book['status'] = "已借出"
        book['borrower'] = user_name
        book['borrow_date'] = datetime.now().strftime("%Y-%m-%d")
        
        return True, f"成功借阅《{book['title']}》"
    
    def return_book(self, book_id):
        """归还图书"""
        if book_id not in self.books_db:
            return False, "图书不存在"
        
        book = self.books_db[book_id]
        if book['status'] != "已借出":
            return False, f"图书《{book['title']}》未被借出"
        
        # 更新状态
        book['status'] = "可借阅"
        if 'borrower' in book:
            del book['borrower']
        if 'borrow_date' in book:
            del book['borrow_date']
        
        return True, f"成功归还《{book['title']}》"
    
    def get_recommendations(self, user_preference):
        """基于用户偏好推荐图书"""
        # 这里可以实现更复杂的推荐算法
        available_books = self.search_books(status="可借阅")
        
        if user_preference:
            # 根据偏好关键词筛选
            preference_books = self.search_books(keyword=user_preference, status="可借阅")
            if preference_books:
                return preference_books[:3]  # 返回前3本
        
        return available_books[:3]  # 默认返回前3本可借阅的书
    
    def consult(self, user_query, user_id="default"):
        """图书咨询服务"""
        try:
            # 获取格式化的图书信息
            books_info = self._format_books_info()
            
            # 调用咨询链
            response = self.with_msg_hist.invoke(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "books_info": books_info
                },
                config={'configurable': {'session_id': user_id}}
            )
            
            return response
            
        except Exception as e:
            return f" 咨询过程中出现错误：{str(e)}"
    
    def list_books(self):
        """列出所有图书"""
        print("\n 图书库存清单:")
        print("=" * 80)
        
        for book_id, book in self.books_db.items():
            status_icon = "[可借]" if book['status'] == "可借阅" else "[已借]"
            print(f"{status_icon} [{book_id}] 《{book['title']}》")
            print(f"    作者: {book['author']}")
            print(f"    分类: {book['category']}")
            print(f"    状态: {book['status']}")
            print(f"    简介: {book['description'][:50]}...")
            print("-" * 50)
    
    def clear_user_history(self, user_id="default"):
        """清除用户聊天历史"""
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
            print(f" 已清除用户 {user_id} 的咨询历史")
        else:
            print(f" 用户 {user_id} 没有咨询历史")


def main():
    """主函数 - 交互式图书管理界面"""
    print(" 欢迎使用智能图书管理系统！")
    print("=" * 60)
    
    # 创建图书管理系统实例
    library = BookManagementSystem()
    
    print(" 系统功能:")
    print("  1. 图书咨询 - 直接提问关于图书的任何问题")
    print("  2. 图书搜索 - 输入 'search [关键词]'")
    print("  3. 查看所有图书 - 输入 'list'")
    print("  4. 借阅图书 - 输入 'borrow [书籍ID] [姓名]'")
    print("  5. 归还图书 - 输入 'return [书籍ID]'")
    print("  6. 获取推荐 - 输入 'recommend [偏好]'")
    print("  7. 清除历史 - 输入 'clear'")
    print("  8. 退出系统 - 输入 'quit'")
    print("=" * 60)
    
    # 显示图书清单
    library.list_books()
    
    while True:
        user_input = input("\n 请输入您的需求: ").strip()
        
        if user_input.lower() == 'quit':
            print(" 感谢使用图书管理系统，再见！")
            break
        
        elif user_input.lower() == 'list':
            library.list_books()
        
        elif user_input.lower() == 'clear':
            library.clear_user_history()
        
        elif user_input.lower().startswith('search '):
            keyword = user_input[7:].strip()
            results = library.search_books(keyword=keyword)
            
            if results:
                print(f"\n 搜索结果 (关键词: {keyword}):")
                for book_id, book in results:
                    status_icon = "[可借]" if book['status'] == "可借阅" else "[已借]"
                    print(f"{status_icon} [{book_id}] 《{book['title']}》- {book['author']}")
                    print(f"    简介: {book['description'][:100]}...")
            else:
                print(f" 没有找到包含关键词 '{keyword}' 的图书")
        
        elif user_input.lower().startswith('borrow '):
            parts = user_input[7:].strip().split(' ', 1)
            if len(parts) >= 2:
                book_id, user_name = parts[0], parts[1]
                success, message = library.borrow_book(book_id, user_name)
                print(f"{'[成功]' if success else '[失败]'} {message}")
            else:
                print(" 请提供书籍ID和借阅人姓名，格式：borrow [书籍ID] [姓名]")
        
        elif user_input.lower().startswith('return '):
            book_id = user_input[7:].strip()
            success, message = library.return_book(book_id)
            print(f"{'[成功]' if success else '[失败]'} {message}")
        
        elif user_input.lower().startswith('recommend'):
            preference = user_input[9:].strip() if len(user_input) > 9 else ""
            recommendations = library.get_recommendations(preference)
            
            if recommendations:
                print(f"\n 为您推荐以下图书:")
                for book_id, book in recommendations:
                    print(f"[推荐] [{book_id}] 《{book['title']}》- {book['author']}")
                    print(f"    推荐理由: {book['description'][:80]}...")
                    print(f"    适合读者: {book['suitable_readers']}")
            else:
                print(" 暂无可推荐的图书")
        
        elif user_input:
            # 图书咨询
            print("\n 图书管理员回复:")
            response = library.consult(user_input)
            print(response)
        
        else:
            print(" 请输入有效内容")


if __name__ == '__main__':
    main() 