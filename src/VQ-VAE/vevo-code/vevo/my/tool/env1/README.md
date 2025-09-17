# VEVO环境配置工具

此目录包含了配置VEVO项目运行环境所需的所有工具和文档。

## 📁 文件说明

### 依赖配置文件
- **`vevo_complete_requirements.txt`** - 完整的依赖列表，包含所有需要的Python包 (已更新)
- **`vevo_exact_requirements.txt`** - 基于用户提供的精确依赖列表
- **`requirements_minimal.txt`** - 最小依赖配置，适合快速测试
- **`missing_packages_analysis.txt`** - 依赖对比分析文档

### 自动化安装脚本
- **`setup_vevo_environment.sh`** - Linux/macOS环境自动配置脚本
- **`setup_vevo_environment.bat`** - Windows环境自动配置脚本

### 文档
- **`VEVO环境配置指南.md`** - 详细的环境配置说明文档

## 🚀 快速使用

### 1. 自动化安装 (推荐)

**Linux/macOS用户:**
```bash
cd models/vc/vevo/my/tool/env1
chmod +x setup_vevo_environment.sh
./setup_vevo_environment.sh
```

**Windows用户:**
```cmd
cd models\vc\vevo\my\tool\env1
setup_vevo_environment.bat
```

### 2. 手动安装

**最小安装 (快速测试):**
```bash
pip install -r requirements_minimal.txt
```

**完整安装:**
```bash
pip install -r vevo_complete_requirements.txt
```

**精确安装 (基于用户提供的列表):**
```bash
pip install -r vevo_exact_requirements.txt
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
```

## ⚠️ 重要提醒

1. **系统依赖**: 确保已安装 `espeak-ng` 和 `ffmpeg`
2. **Python版本**: 需要 Python >= 3.9
3. **虚拟环境**: 建议使用虚拟环境避免包冲突

## 📖 详细说明

请查看 `VEVO环境配置指南.md` 获取完整的配置说明和故障排除指南。

## 🔧 自定义配置

如需修改依赖版本或添加新的包，请编辑对应的requirements文件。

---

**位置**: `models/vc/vevo/my/tool/env1/`  
**用途**: VEVO项目环境配置  
**更新**: 根据项目需求定期更新依赖版本
