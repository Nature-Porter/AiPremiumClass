# main.py
from library_ai import LibrarianAI

def main():
    print("--- 欢迎来到智图AI图书馆服务 ---")
    print("我是您的智能图书管理员，可以帮您借还书、推荐图书。")
    print("您可以试试说：")
    print("  - '我想借《三体》'")
    print("  - '帮我推荐几本科幻小说'")
    print("  - '我正在读的书有哪些？'")
    print("  - '我想还《百年孤独》这本书，它的ID是B003'")
    print("输入 'exit' 退出程序。")
    print("-" * 30)

    # 我们假设当前登录的用户是 张伟 (U12345)
    ai_librarian = LibrarianAI(user_id="U12345")
    
    while True:
        user_input = input("您好，请问有什么可以帮您？\n> ")
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("感谢您的使用，再见！")
            break
        
        response = ai_librarian.chat(user_input)
        print(f"\n智图AI: {response}\n")

if __name__ == "__main__":
    main()