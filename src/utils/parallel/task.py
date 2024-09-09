import importlib.util
import sys
import os

# 假设你的函数所在的文件路径为 "path/to/your_module.py"
file_path = "../../../target.py"
module_name = "target"
sys.path.append(os.path.dirname(file_path))

# 加载模块
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 现在可以调用模块中的函数
module.target({}, gpu=1, path=None)  # 这里调用你需要运行的函数
