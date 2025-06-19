import  os
from dotenv  import load_dotenv,find_dotenv
from openai import OpenAI
# path = find_dotenv()

# load_dotenv(path)

# print(os.environ['api_key'])
# print(os.environ['base_url'])

if __name__ =="__main__":

    load_dotenv(find_dotenv())
    client = OpenAI(
        api_key =os.environ["api_key"],
        base_url =os.environ["base_url"]
    )
    
    response = client.chat.completions.create(
       model ="glm-4-flash",
       messages =[
           {
               'role':'user','content':'今天是6月19日，请列举历史上今天发生过的十件大事'
           } 
       ]
    )
    print(response.choices[0].message.content)


