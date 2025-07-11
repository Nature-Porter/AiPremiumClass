"""
Agent实现PDF和TAVILY工具组合动态调用
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
import os

def setup_embedding_model():
    """配置embedding模型，支持多种方案"""
    print("正在配置embedding模型...")
    
    try:
        # 方案1：尝试使用智谱AI专用的embedding类
        from langchain_community.embeddings import ZhipuAIEmbeddings
        print("使用智谱AI专用embedding类")
        embedding_model = ZhipuAIEmbeddings(
            api_key=os.environ['api_key_e']
        )
        print("方案1成功：使用智谱AI专用embedding")
        return embedding_model
    except (ImportError, KeyError):
        print("智谱AI专用类不可用，尝试OpenAI格式")
        
        try:
            # 方案2：使用OpenAI格式（支持不同服务）
            api_key = os.environ.get('OPENAI_API_KEY', os.environ.get('api_key'))
            base_url = os.environ.get('BASE_URL', os.environ.get('base_url'))
            
            if api_key and base_url:
                print("使用OpenAI格式embedding")
                embedding_model = OpenAIEmbeddings(
                    api_key=api_key,
                    base_url=base_url
                )
                print("方案2成功：使用OpenAI格式embedding")
                return embedding_model
        except Exception as e:
            print(f"OpenAI格式embedding失败: {e}")
        
        # 方案3：使用免费的Hugging Face embedding模型（备用方案）
        print("使用免费的Hugging Face embedding模型")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("方案3成功：使用Hugging Face免费模型")
        return embedding_model

def setup_pdf_retriever(pdf_path, chroma_db_path):
    """设置PDF检索器"""
    # 配置embedding模型
    embedding_model = setup_embedding_model()
    
    if not os.path.exists(chroma_db_path):
        print("正在加载PDF文档...")
        # 使用PDFMinerLoader加载PDF文件
        try:
            loader = PDFMinerLoader(pdf_path)
            docs = loader.load()
            print(f"已加载 {len(docs)} 个文档页面")
        except Exception as e:
            print(f"加载PDF文件失败: {e}")
            raise e

        # TextSplitter实现加载后Document分割
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '. ', ' '],
            chunk_size=1000,
            chunk_overlap=200
        )
        splited_docs = splitter.split_documents(docs)
        print(f"文档已分割为 {len(splited_docs)} 个块")
        
        # 创建Chroma向量数据库（持久化存储）
        print("正在创建向量数据库...")
        vector_store = Chroma.from_documents(
            documents=splited_docs,
            embedding=embedding_model,
            persist_directory=chroma_db_path
        )
        print('PDF Chroma向量数据库创建并保存成功！')
    else:
        print("加载现有的PDF向量数据库...")
        vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embedding_model
        )
        print('PDF Chroma向量数据库加载成功！')

    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def setup_tavily_search():
    """设置Tavily搜索工具，并处理API密钥"""
    # 检查Tavily API密钥
    tavily_api_key = os.environ.get('TAVILY_API_KEY')
    if not tavily_api_key:
        print("警告: 未找到TAVILY_API_KEY环境变量")
        print("请输入Tavily API密钥 (或按回车跳过Tavily搜索功能): ")
        user_input = input()
        if user_input.strip():
            os.environ['TAVILY_API_KEY'] = user_input
            print("已设置Tavily API密钥")
            return TavilySearchResults(max_results=2)
        else:
            print("未提供Tavily API密钥，将禁用Tavily搜索功能")
            return None
    else:
        print("已找到Tavily API密钥")
        return TavilySearchResults(max_results=2)

def main():
    load_dotenv()
    
    # 构建LLM
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']) # 支持function call
    
    # 提示词模版
    prompt = hub.pull("hwchase17/openai-functions-agent")
    
    # 构建工具
    # 1. Tavily搜索工具
    search = setup_tavily_search()
    
    # 2. PDF检索工具
    pdf_path = "week15/documents/era_of_experience.pdf"
    chroma_db_path = "week15/pdf_chroma_db"
    
    # 确保PDF文件存在
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return
    
    pdf_retriever = setup_pdf_retriever(pdf_path, chroma_db_path)
    pdf_tool = create_retriever_tool(
        pdf_retriever, 
        "pdf_retriever",
        description="检索关于'The Era of Experience'论文的内容，这是一篇由DeepMind发布的关于AI体验时代的论文，讨论了AI如何影响人类体验、面临的挑战以及可能的解决方案。"
    )
    
    # 工具列表
    tools = []
    if search:
        tools.append(search)
    tools.append(pdf_tool)
    
    # 构建agent
    agent = create_tool_calling_agent(
        llm=model, prompt=prompt, tools=tools)
    
    # agent executor
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True)
    
    # 运行agent
    print("\n=== Agent工具调用测试 ===")
    
    # 测试问题列表
    questions = [
        "什么是Experience时代?",
        "请结合最新的AI研究进展，分析AI如何影响人类体验",
        "DeepMind在Era of Experience论文中提出了哪些解决方案？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)
        try:
            msgs = executor.invoke({"input": question})
            print(f"回答: {msgs['output']}")
        except Exception as e:
            print(f"错误: {e}")
        print("=" * 50)

if __name__ == "__main__":
    main() 