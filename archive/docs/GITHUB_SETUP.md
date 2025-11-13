# GitHub 发布准备指南

本文档说明如何将此项目发布到GitHub。

## ✅ 已完成的规范化工作

### 1. 标准GitHub项目文件
- ✅ `.gitignore` - 忽略不必要的文件
- ✅ `LICENSE` - MIT许可证
- ✅ `setup.py` - Python包安装配置
- ✅ `requirements.txt` - 依赖管理
- ✅ `README.md` - 英文主文档（GitHub友好）
- ✅ `README_ZH.md` - 中文详细文档
- ✅ `CITATION.bib` - 学术引用
- ✅ `CONTRIBUTING.md` - 贡献指南

### 2. 项目结构规范化
- ✅ 核心代码保留在主目录
- ✅ 归档文件移至 `archive/`
- ✅ Slurm脚本移至 `scripts/`
- ✅ 示例代码移至 `examples/`
- ✅ 删除重复和废弃文件
- ✅ 清理Python缓存

### 3. 文档完善
- ✅ `PROJECT_STRUCTURE.md` - 项目结构总览
- ✅ `CORE_STRUCTURE.md` - 核心代码详解
- ✅ `DEPRECATED.md` - 废弃文件说明
- ✅ `examples/README.md` - 示例说明
- ✅ `scripts/README.md` - 脚本说明

### 4. 代码组织
- ✅ `__init__.py` - 包初始化
- ✅ 清晰的模块划分（datasets/, evaluation/）
- ✅ 统一的配置管理（config.py）

## 📋 发布到GitHub的步骤

### 步骤1: 创建GitHub仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - Repository name: `emotion-rvq-bottleneck`
   - Description: `Emotion Recognition Robustness under RVQ Information Bottleneck with Entropy Coding`
   - 选择 Public（如果要开源）或 Private
   - **不要**勾选"Initialize with README"（我们已有README.md）
   - **不要**添加.gitignore或license（我们已有）

### 步骤2: 初始化Git仓库

```bash
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 初始化git仓库
git init

# 添加所有文件
git add .

# 首次提交
git commit -m "Initial commit: Emotion RVQ Bottleneck project v1.0.0"
```

### 步骤3: 连接到GitHub并推送

```bash
# 添加远程仓库（替换为您的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/emotion-rvq-bottleneck.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

### 步骤4: 配置GitHub仓库（在网页上）

1. **添加Topics**:
   - `emotion-recognition`
   - `vector-quantization`
   - `entropy-coding`
   - `information-bottleneck`
   - `pytorch`
   - `speech-processing`

2. **设置Description**:
   ```
   Emotion Recognition Robustness under RVQ Information Bottleneck with Entropy Coding
   ```

3. **启用Issues**（可选）

4. **添加Repository Details**:
   - Website: 您的论文链接或个人主页
   - Tags: 如上述topics

## 🔧 发布前需要修改的内容

### 必须修改的文件

#### 1. `LICENSE`
```diff
- Copyright (c) 2025 [Your Name/Institution]
+ Copyright (c) 2025 您的姓名/机构名称
```

#### 2. `setup.py`
```diff
- author="Your Name",
- author_email="your.email@example.com",
- url="https://github.com/yourusername/emotion-rvq-bottleneck",
+ author="您的姓名",
+ author_email="您的邮箱",
+ url="https://github.com/您的用户名/emotion-rvq-bottleneck",
```

#### 3. `CITATION.bib`
```diff
- author={Your Name and Collaborators},
- journal={arXiv preprint arXiv:XXXX.XXXXX},
+ author={您的姓名 and 合作者},
+ journal={会议/期刊名称},
```

#### 4. `README.md`
```diff
- Maintainer: Your Name (your.email@example.com)
- Issues: [GitHub Issues](https://github.com/yourusername/...)
+ Maintainer: 您的姓名 (您的邮箱)
+ Issues: [GitHub Issues](https://github.com/您的用户名/...)
```

#### 5. `CONTRIBUTING.md`
```diff
- 项目维护者: Your Name (your.email@example.com)
- Issue Tracker: https://github.com/yourusername/...
+ 项目维护者: 您的姓名 (您的邮箱)
+ Issue Tracker: https://github.com/您的用户名/...
```

### 可选修改

#### 添加Badges到README.md

在README.md顶部添加更多badges（论文发表后）:

```markdown
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Conference](https://img.shields.io/badge/Conference-ACL2025-blue.svg)](您的论文链接)
```

## 📦 可选：发布到PyPI

如果希望用户可以通过 `pip install emotion-rvq-bottleneck` 安装：

```bash
# 安装发布工具
pip install build twine

# 构建
python -m build

# 上传到PyPI（需要PyPI账号）
twine upload dist/*
```

## ⚠️ 注意事项

### 不应上传的文件（.gitignore已配置）

- ❌ 训练好的模型文件（*.pt, *.pth）- 太大，使用Git LFS或单独托管
- ❌ 数据文件（*.npy, *.npz, *.wav）- 太大
- ❌ 评估结果（evaluation_results/）- 可选
- ❌ 缓存文件（indices_cache/, __pycache__/）
- ❌ Slurm输出（slurm-*.out）

### 例外：保留的重要文件

- ✅ `ev2_mean_std_100h_EN_ZH.npz` - 归一化参数（小文件）
- ✅ `checkpoints/grouped_rvq_best.pt` - 最佳模型（可选，如果<100MB）
- ✅ `checkpoints/entropy_model_best.pt` - 最佳熵模型（可选）

**建议**: 大模型文件使用以下方式共享：
- [Hugging Face Hub](https://huggingface.co/) - 推荐
- [Google Drive](https://drive.google.com/)
- [Zenodo](https://zenodo.org/) - 学术数据托管

### 模型文件托管（推荐）

在README.md中添加模型下载链接：

```markdown
## 📥 Pre-trained Models

Download pre-trained checkpoints:
- [Grouped RVQ Model](https://huggingface.co/yourusername/emotion-rvq/blob/main/grouped_rvq_best.pt) (XXX MB)
- [Entropy Model](https://huggingface.co/yourusername/emotion-rvq/blob/main/entropy_model_best.pt) (XXX MB)

Place them in the `checkpoints/` directory.
```

## 🚀 发布后的工作

### 1. 添加GitHub Actions（可选）

创建 `.github/workflows/tests.yml` 用于自动测试。

### 2. 创建Release

当论文被接收后：

```bash
git tag -a v1.0.0 -m "Version 1.0.0 - Initial release"
git push origin v1.0.0
```

在GitHub上创建Release，附上：
- 论文链接
- 模型下载链接
- 变更日志

### 3. 更新文档

论文发表后更新：
- README.md中的论文链接
- CITATION.bib中的正式引用
- 添加arXiv或会议badges

## 📝 Git使用建议

### 常用命令

```bash
# 查看状态
git status

# 添加新文件
git add filename.py

# 提交更改
git commit -m "描述更改内容"

# 推送到GitHub
git push

# 拉取更新
git pull

# 创建分支
git checkout -b feature/new-feature
```

### .gitignore已配置

不用担心误提交缓存或大文件，`.gitignore`已经配置好了。

## ✅ 检查清单

发布前请确认：

- [ ] 修改LICENSE中的版权信息
- [ ] 修改setup.py中的作者和URL
- [ ] 修改CITATION.bib中的作者信息
- [ ] 修改README.md和CONTRIBUTING.md中的联系方式
- [ ] 检查.gitignore是否正确
- [ ] 删除或归档不必要的文件
- [ ] 测试examples/中的示例代码
- [ ] 准备好模型文件的托管方案（Hugging Face等）
- [ ] 准备好数据集的访问说明

## 🎉 完成！

完成以上步骤后，您的项目就可以成功发布到GitHub了！

---

如有问题，请参考：
- [GitHub官方文档](https://docs.github.com/)
- [Git教程](https://git-scm.com/book/zh/v2)

