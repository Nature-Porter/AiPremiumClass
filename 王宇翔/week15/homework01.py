from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFMinerLoader
import requests
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # 使用Chroma替代FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain import hub

import os
import shutil

def download_pdf(url, local_path):
    """下载PDF文件到本地"""
    if not os.path.exists(local_path):
        print(f"正在下载PDF文件: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF文件已下载到: {local_path}")
        except Exception as e:
            print(f"下载PDF文件失败: {e}")
            return False
    else:
        print(f"PDF文件已存在: {local_path}")
    return True

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
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("方案3成功：使用Hugging Face免费模型")
        return embedding_model

def setup_llm_model():
    """配置LLM模型，支持多种方案"""
    print("正在配置LLM模型...")
    
    try:
        # 优先使用智谱AI
        api_key = os.environ.get('api_key')
        base_url = os.environ.get('base_url')
        
        if api_key and base_url:
            print("使用智谱AI GLM模型")
            return ChatOpenAI(
                model="glm-4-flash-250414",
                base_url=base_url,
                api_key=api_key,
            )
    except Exception as e:
        print(f"智谱AI模型配置失败: {e}")
    
    try:
        # 备用：使用OpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        base_url = os.environ.get('BASE_URL', 'https://api.openai.com/v1')
        
        if api_key:
            print("使用OpenAI GPT模型")
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=api_key,
                base_url=base_url
            )
    except Exception as e:
        print(f"OpenAI模型配置失败: {e}")
        
    raise RuntimeError("无法配置任何LLM模型，请检查API配置")

if __name__ == '__main__':
    load_dotenv()
    
    # PDF文档URL和本地路径
    pdf_url = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
    pdf_path = "week15/documents/era_of_experience.pdf"
    
    # 下载PDF文件
    if not download_pdf(pdf_url, pdf_path):
        print("无法下载PDF文件，程序退出")
        exit(1)
    
    # 配置模型
    llm = setup_llm_model()
    embedding_model = setup_embedding_model()
    
    # 定义Chroma数据库存储路径
    chroma_db_path = "week15/pdf_chroma_db"

    if not os.path.exists(chroma_db_path):
        print("正在加载PDF文档...")
        # 使用PDFMinerLoader加载PDF文件
        try:
            loader = PDFMinerLoader(pdf_path)
            docs = loader.load()
            print(f"已加载 {len(docs)} 个文档页面")
        except Exception as e:
            print(f"加载PDF文件失败: {e}")
            exit(1)

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

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    # prompt 
    prompt = hub.pull('rlm/rag-prompt')

    # 构建rag chain
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # 测试问题列表
    questions = [
        "What is the Era of Experience?",
        "What are the main challenges discussed in this paper?", 
        "How does AI impact human experience according to this document?",
        "What solutions are proposed in this paper?"
    ]

    print("\n=== PDF RAG 问答系统测试 ===")
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)
        try:
            response = rag_chain.invoke(question)
            print(f"回答: {response}")
        except Exception as e:
            print(f"错误: {e}")
        print("=" * 50) 