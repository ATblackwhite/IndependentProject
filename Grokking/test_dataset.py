from datasets import AlgorithmicDataset
from binary_operations import add_square_mod

def test_algorithmic_dataset():
    
    # 创建数据集实例，使用模数5
    dataset = AlgorithmicDataset(add_square_mod, p=97)
    
    print(f"数据集大小: {len(dataset)}")
    print("\n前10个样本:")
    for i in range(min(10, len(dataset))):
        inputs, target = dataset[i]
        print(f"样本 {i}:")
        print(f"输入: {inputs}")
        print(f"目标值: {target}")
        print("-" * 50)

if __name__ == "__main__":
    test_algorithmic_dataset() 