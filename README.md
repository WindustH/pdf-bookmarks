# PDF Bookmarks

使用视觉 LLM 自动为 PDF 电子书添加书签的工具。通过分析目录，自动生成书签并应用到 PDF 文件中。

## 功能特点

- 自动检测 PDF 中的目录页
- 使用视觉 LLM 提取书签信息
- 自动计算页码偏移量
- 支持多层级书签结构
- 使用文本 LLM 优化书签格式
- 基于 pdftk 的书签应用

## 平台安装指南

### Linux

#### 1. 安装 Python

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora/RHEL
sudo dnf install python3 python3-pip

# Arch Linux
sudo pacman -S python python-pip
```

#### 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或使用 pip：

```bash
pip install uv
```

#### 3. 安装 pdftk

```bash
# Ubuntu/Debian
sudo apt install pdftk

# Fedora/RHEL
sudo dnf install pdftk

# Arch Linux
sudo pacman -S pdftk
```

#### 4. 安装项目依赖

```bash
cd /path/to/pdf-bookmarks
uv sync
```

### macOS

#### 1. 安装 Python

使用 [Homebrew](https://brew.sh/)：

```bash
brew install python@3.12
```

或下载官方安装包：[python.org](https://www.python.org/downloads/)

#### 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或使用 Homebrew：

```bash
brew install uv
```

#### 3. 安装 pdftk

```bash
brew install pdftk-java
```

#### 4. 安装项目依赖

```bash
cd /path/to/pdf-bookmarks
uv sync
```

### Windows

#### 1. 安装 Python

1. 访问 [python.org](https://www.python.org/downloads/)
2. 下载 Python 3.10+ 安装包
3. 运行安装程序，**务必勾选 "Add Python to PATH"**

或使用 [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)：

```powershell
winget install Python.Python.3.12
```

#### 2. 安装 uv

使用 PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

或下载独立的 uv 可执行文件：[github.com/astral-sh/uv](https://github.com/astral-sh/uv/releases)

#### 3. 安装 pdftk

1. 下载 [pdftk-server](https://www.pdflabs.com/tools/pdftk-server/)
2. 运行安装程序
3. 将 pdftk 添加到系统 PATH（安装程序通常会自动处理）

或使用 [Chocolatey](https://chocolatey.org/)：

```powershell
choco install pdftk
```

#### 4. 安装项目依赖

打开 Command Prompt 或 PowerShell：

```cmd
cd C:\path\to\pdf-bookmarks
uv sync
```

## 配置

在项目根目录创建 `model.env` 文件，配置以下环境变量：

```env
API_KEY=your_api_key_here
BASE_URL=https://api.example.com/v1
VISION_MODEL=gpt-4-vision-preview
TEXT_MODEL=gpt-4
TOC_WORKERS=4
REFINE_TIMEOUT=600
```

## 使用方法

### 基本使用

```bash
# 所有平台
uv run python src/main.py input.pdf output.pdf
```

### 完整示例

```bash
# Linux / macOS
uv run python src/main.py ~/Documents/ebook.pdf ~/Documents/ebook_with_bookmarks.pdf

# Windows
uv run python src/main.py C:\Users\YourName\Documents\ebook.pdf C:\Users\YourName\Documents\ebook_with_bookmarks.pdf
```

## 工作原理

1. **扫描 TOC 页**：逐页扫描 PDF，使用视觉 LLM 识别目录页
2. **计算页码偏移**：找到第一个条目的页码，在实际 PDF 中定位其内容，计算偏移量
3. **提取书签信息**：独立并行处理目录页（`TOC_WORKERS`，默认 4），不传入前页上下文；逐页保存结果并按原页序合并
4. **优化书签**：文本 LLM 通过 `replace_text` tool call 精确修改局部原文，通过 `finish_refinement` 完成检查，无需重新输出全文
5. **应用偏移量**：根据计算的偏移量调整页码
6. **生成 PDF**：使用 pdftk 将书签应用到 PDF

## 常见问题

### pdftk 命令未找到

确保 pdftk 已正确安装并添加到系统 PATH：

```bash
# Linux / macOS
which pdftk

# Windows
where pdftk
```


流式文本达到输出长度限制时，自动携带已输出内容续写，最多续写 20 次；连接异常或未收到完成标志时不将半截内容视为成功。并行提取期间仅显示每页完成状态，避免流式文字交错。文本模型及其 API 需支持 Chat Completions function calling。

断点续跑会复用已保存的逐页结果，只重试缺失页；旧版仅保存合并文本的提取进度会重新提取目录页。

运行离线回归测试：`uv run python -m unittest discover -s tests -v`。

refine 使用流式 tool call，收到完整参数与结束标志后才执行修改。`REFINE_TIMEOUT` 控制单次网络读取的等待上限（秒，默认 600），不是整个 refine 阶段的总时限；有持续流数据时可继续运行。服务端或代理的超时限制仍可能提前终止请求。
