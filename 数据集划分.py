# -*- coding: utf-8 -*-
import os
import shutil
import random

def split_dataset(voice_set_dir="./voice_set", 
                  train_dir="./processed_train_records",
                  test_dir="./processed_test_records",
                  train_ratio=0.8):
    """
    将voice_set数据集划分为训练集和测试集，并按照标准命名格式重命名
    
    参数:
    voice_set_dir: 原始voice_set目录路径
    train_dir: 训练集输出目录
    test_dir: 测试集输出目录
    train_ratio: 训练集比例，默认0.8 (4:1)
    """
    
    # 创建输出目录
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # 创建数字子目录
    for i in range(10):
        train_digit_dir = os.path.join(train_dir, f"digit_{i}")
        test_digit_dir = os.path.join(test_dir, f"digit_{i}")
        if not os.path.exists(train_digit_dir):
            os.makedirs(train_digit_dir)
        if not os.path.exists(test_digit_dir):
            os.makedirs(test_digit_dir)
    
    # 统计数据信息
    total_files = 0
    for digit in range(10):
        digit_dir = os.path.join(voice_set_dir, str(digit))
        if os.path.exists(digit_dir):
            files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            total_files += len(files)
            print(f"数字 {digit}: {len(files)} 个文件")
    
    print(f"\n总共发现 {total_files} 个音频文件")
    
    # 处理每个数字类别
    for digit in range(10):
        print(f"\n正在处理数字 {digit}...")
        
        digit_dir = os.path.join(voice_set_dir, str(digit))
        # digit_dir=os.path.join(voice_set_dir, "digit_"+str(digit))
        if not os.path.exists(digit_dir):
            print(f"警告: 目录 {digit_dir} 不存在，跳过")
            continue
        
        # 获取所有wav文件
        files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
        
        if not files:
            print(f"警告: 数字 {digit} 目录中没有wav文件，跳过")
            continue
        
        # 随机打乱文件顺序
        random.shuffle(files)
        
        # 计算训练集和测试集的分割点
        split_point = int(len(files) * train_ratio)
        
        train_files = files[:split_point]
        test_files = files[split_point:]
        
        print(f"  训练集: {len(train_files)} 个文件")
        print(f"  测试集: {len(test_files)} 个文件")
        
        # 复制并重命名训练集文件
        for idx, filename in enumerate(train_files, 1):
            src_path = os.path.join(digit_dir, filename)
            dst_filename = f"{idx}_{digit}.wav"
            dst_path = os.path.join(train_dir, f"digit_{digit}", dst_filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                print(f"    训练集: {filename} -> {dst_filename}")
            except Exception as e:
                print(f"    错误: 无法复制 {src_path} 到 {dst_path}: {e}")
        
        # 复制并重命名测试集文件
        for idx, filename in enumerate(test_files, len(train_files) + 1):
            src_path = os.path.join(digit_dir, filename)
            dst_filename = f"{idx}_{digit}.wav"
            dst_path = os.path.join(test_dir, f"digit_{digit}", dst_filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                print(f"    测试集: {filename} -> {dst_filename}")
            except Exception as e:
                print(f"    错误: 无法复制 {src_path} 到 {dst_path}: {e}")
    
    print(f"\n数据集划分完成!")
    print(f"训练集保存至: {train_dir}")
    print(f"测试集保存至: {test_dir}")

def verify_dataset_split(train_dir="./processed_train_records", 
                        test_dir="./processed_test_records"):
    """
    验证数据集划分结果
    """
    print("\n验证数据集划分结果:")
    print("=" * 50)
    
    for digit in range(10):
        train_digit_dir = os.path.join(train_dir, f"digit_{digit}")
        test_digit_dir = os.path.join(test_dir, f"digit_{digit}")
        
        train_files = []
        test_files = []
        
        if os.path.exists(train_digit_dir):
            train_files = [f for f in os.listdir(train_digit_dir) if f.endswith('.wav')]
        
        if os.path.exists(test_digit_dir):
            test_files = [f for f in os.listdir(test_digit_dir) if f.endswith('.wav')]
        
        print(f"数字 {digit}: 训练集 {len(train_files)} 个文件, 测试集 {len(test_files)} 个文件")
        
        # 验证文件命名格式
        for file_list, set_name in [(train_files, "训练集"), (test_files, "测试集")]:
            for filename in file_list:
                # 检查文件名格式: {数字}_{数字}.wav
                expected_suffix = f"_{digit}.wav"
                if not filename.endswith(expected_suffix):
                    print(f"  警告: {set_name} 文件 {filename} 命名格式不正确")
                
                # 检查序号是否连续
                try:
                    file_num = int(filename.split('_')[0])
                except:
                    print(f"  警告: {set_name} 文件 {filename} 序号格式不正确")

if __name__ == '__main__':
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 划分数据集
    split_dataset(
        voice_set_dir="./voice_data",
        train_dir="./processed_train_records", 
        test_dir="./processed_test_records",
        train_ratio=0.8  # 4:1 比例
    )
    
    # 验证划分结果
    verify_dataset_split()
    
    print("\n数据集准备完成，可以开始训练模型!")