import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import time

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv(find_dotenv())
    
    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url=os.environ["base_url"]
    )
    
    # 使用更合理的temperature值范围(0.0-1.0)
    for temperature_ in [0.0, 0.2, 0.5, 0.8, 1.0]:
        print(f"\n=== Temperature: {temperature_} ===")
        
        try:
            # 修改提示语，使用更加中性的内容
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {
                        'role': 'user',
                        'content': '请列举五个著名的科学发现及其年份'
                    } 
                ],
                temperature=temperature_,
                max_tokens=500,
            )
            
            print(response.choices[0].message.content)
        except Exception as e:
            print(f"发生错误: {e}")
            # 添加短暂延迟，避免频繁请求
            time.sleep(2)