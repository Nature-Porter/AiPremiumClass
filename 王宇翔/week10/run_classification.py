import os
import subprocess
import argparse

def install_requirements():
    """安装所需的依赖包"""
    requirements = [
        "torch",
        "transformers",
        "pandas",
        "openpyxl",
        "scikit-learn",
        "tensorboard",
        "numpy"
    ]
    
    for req in requirements:
        print(f"安装 {req}...")
        subprocess.call(["pip", "install", req])
    
    print("依赖包安装完成！")

def run_training(epochs=5, batch_size=16, learning_rate=2e-5):
    """运行模型训练"""
    from jd_comment_classification import main
    
    # 设置环境变量
    os.environ["EPOCHS"] = str(epochs)
    os.environ["BATCH_SIZE"] = str(batch_size)
    os.environ["LEARNING_RATE"] = str(learning_rate)
    
    # 运行主程序
    main()
    
    # 打印TensorBoard启动命令
    print("\n要查看训练过程的可视化结果，请运行以下命令：")
    print("tensorboard --logdir=logs")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="京东评论文本分类训练脚本")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--install_deps", action="store_true", help="是否安装依赖")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.install_deps:
        install_requirements()
    
    run_training(args.epochs, args.batch_size, args.learning_rate) 