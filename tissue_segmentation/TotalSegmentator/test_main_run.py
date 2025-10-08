#!/usr/bin/env python3
"""
测试 main_run.py 脚本的功能
"""
import subprocess
import sys
from pathlib import Path

def test_main_run():
    """测试 main_run.py 的基本功能"""
    
    print("🧪 测试 main_run.py 脚本")
    print("=" * 40)
    
    # 1. 测试列出模型
    print("\n1. 测试列出可用模型...")
    try:
        result = subprocess.run([
            sys.executable, "main_run.py", "--list-models"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ 列出模型功能正常")
            print("可用模型:")
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('='):
                    print(f"  {line}")
        else:
            print(f"❌ 列出模型失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 2. 测试帮助信息
    print("\n2. 测试帮助信息...")
    try:
        result = subprocess.run([
            sys.executable, "main_run.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ 帮助信息正常")
        else:
            print(f"❌ 帮助信息失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 3. 测试参数验证
    print("\n3. 测试参数验证...")
    try:
        # 测试缺少必需参数
        result = subprocess.run([
            sys.executable, "main_run.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0 and "必须指定模型" in result.stdout:
            print("✅ 参数验证正常")
        else:
            print(f"❌ 参数验证异常: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！")
    print("\n现在您可以使用以下命令:")
    print("python main_run.py --list-models")
    print("python main_run.py -m total_6mm -i /path/to/dicom -o result.h5")
    
    return True

if __name__ == "__main__":
    test_main_run()
