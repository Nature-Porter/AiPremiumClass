class MockLMS:
    """一个模拟的图书管理系统，用于存储和管理图书与读者数据。"""
    def __init__(self):
        self.books = {
            "B001": {"title": "三体", "author": "刘慈欣", "genre": "科幻", "status": "available", "tags": ["宏大叙事", "硬核科幻"]},
            "B002": {"title": "流浪地球", "author": "刘慈欣", "genre": "科幻", "status": "available", "tags": ["末日", "脑洞"]},
            "B003": {"title": "百年孤独", "author": "加西亚·马尔克斯", "genre": "魔幻现实主义", "status": "borrowed", "tags": ["家族史诗", "孤独"]},
            "B004": {"title": "基地", "author": "艾萨克·阿西莫夫", "genre": "科幻", "status": "available", "tags": ["银河帝国", "心理史学"]},
            "B005": {"title": "沙丘", "author": "弗兰克·赫伯特", "genre": "科幻", "status": "available", "tags": ["太空歌剧", "生态学"]},
            "B006": {"title": "仿生人会梦见电子羊吗？", "author": "菲利普·迪克", "genre": "科幻", "status": "available", "tags": ["赛博朋克", "哲学思辨"]},
        }
        self.users = {
            "U12345": {
                "name": "张伟",
                "borrowed_books": ["B003"],
                "reading_history": [
                    {"title": "三体", "rating": 5},
                    {"title": "福尔摩斯探案全集", "rating": 4}
                ],
                "preferences": ["科幻", "悬疑"]
            }
        }

    def get_book_info(self, book_id):
        return self.books.get(book_id)

    def get_user_info(self, user_id):
        return self.users.get(user_id)
        
    def get_all_books_for_recommendation(self):
        """为推荐提供简化的图书列表"""
        return [{ "book_id": k, **v } for k, v in self.books.items()]

    def borrow_book(self, user_id, book_id):
        if book_id not in self.books:
            return False, "找不到这本书。"
        if self.books[book_id]["status"] != "available":
            return False, "抱歉，这本书已经被借走了。"
        
        self.books[book_id]["status"] = "borrowed"
        self.users[user_id]["borrowed_books"].append(book_id)
        return True, f"《{self.books[book_id]['title']}》借阅成功！"

    def return_book(self, user_id, book_id):
        if book_id not in self.books:
            return False, "系统里没有这本书的记录。"
        if book_id not in self.users[user_id]["borrowed_books"]:
            return False, "您没有借阅过这本书。"
            
        self.books[book_id]["status"] = "available"
        self.users[user_id]["borrowed_books"].remove(book_id)
        return True, f"《{self.books[book_id]['title']}》归还成功！"