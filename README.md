# video-note

二次开发自[xboHodx/video-study-notes-skill](https://github.com/xboHodx/video-study-notes-skill)

这是一个面向 Windows 的本地视频学习笔记工作区，核心能力来自一个精简版 `.agents` skill：

- 输入一个视频 URL 或本地视频文件
- 优先复用已有字幕
- 没有可用字幕时，使用本地 `fast-whisper-model/` 中配置好的 Whisper 模型转写
- 抽取关键帧
- 生成结构化学习笔记 `notes.md`

当前工作区已经内置了一个可加载的 skill：

- `.agents/skills/video-study-notes/`

它是对 `video-study-notes-skill` 的 Windows 精简适配版。

## 目录结构

- `.agents/skills/video-study-notes/`: skill 定义、模板和脚本
- `fast-whisper-model/`: 本地 Whisper 模型目录根目录
- `.agents/skills/video-study-notes/config.json`: skill 本地配置，声明默认模型路径
- `output/`: 每个视频的输出目录
- `requirements-skill.txt`: 核心 Python 依赖

## 环境要求

建议环境：

- Windows
- conda 环境
- Python 3.11
- `uv`
- `ffmpeg`
- NVIDIA GPU + 驱动（如需 GPU 转写）

本项目已经在如下组合上验证过：

- Python: `3.11.15`
- `ctranslate2`: `4.7.1`
- `faster-whisper`: `1.2.1`
- `yt-dlp`: `2026.3.17`

## 1. 创建并激活 conda 环境

示例：

```powershell
conda create -n video-note python=3.11 -y
conda activate video-note
```

确认当前 Python 来自你的 conda 环境：

```powershell
python -c "import sys; print(sys.executable)"
```

## 2. 安装 Python 依赖

先安装 `uv`：

```powershell
pip install uv
```

再安装项目依赖：

```powershell
uv pip install -r requirements-skill.txt
```

如果你想显式指定当前 conda 环境的 Python：

```powershell
uv pip install --python "<YOUR_CONDA_ENV_PYTHON>" -r requirements-skill.txt
```

## 3. 安装 ffmpeg

要求 `ffmpeg` 可直接从命令行调用：

```powershell
ffmpeg -version
```

如果提示找不到命令，需要安装 `ffmpeg` 后把 `ffmpeg` 的路径加到 `PATH`。

## 4. 准备本地 Whisper 模型

本 skill 不在脚本里硬编码模型名，而是读取：

```text
.agents/skills/video-study-notes/config.json
```

当前默认配置为：

```text
fast-whisper-model/faster-whisper-large-v3-turbo
```

也就是：

```text
<workspace_root>\fast-whisper-model\faster-whisper-large-v3-turbo
```

配置文件示例：

```json
{
  "default_model": "fast-whisper-model/faster-whisper-large-v3-turbo"
}
```

这个目录至少应包含类似文件：

- `config.json`
- `model.bin`
- `tokenizer.json`
- `vocabulary.json`
- `preprocessor_config.json`

如果模型目录不存在，转写脚本会失败。

## 5. 安装 GPU 运行时依赖

如果只想用 CPU，可以跳过这一节。

如果要让 `faster-whisper` 在 Windows 上使用 CUDA，除了显卡驱动之外，还需要安装 CUDA 运行时相关库。这里推荐直接用 `uv` 安装 Python wheel 版本：

```powershell
uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12
```

这会安装类似这些 DLL：

- `cublas64_12.dll`
- `cudart64_12.dll`
- `cudnn64_9.dll`
- `nvrtc64_120_0.dll`

这些文件通常会被安装到：

```text
<env>\Lib\site-packages\nvidia\...\bin
```

例如：

```text
<your_conda_env>\Lib\site-packages\nvidia\cublas\bin\cublas64_12.dll
```

## 6. GPU 可用性验证

先检查显卡和驱动：

```powershell
nvidia-smi
```

再验证音频是否真的走 GPU：

```powershell
python ".agents\skills\video-study-notes\subskills\media-transcribe\scripts\transcribe_audio.py" --input <input> --output-dir <output>
```

如果命令行回显没有出现 `CUDA transcription failed (Library cublas64_12.dll is not found or cannot be loaded). Falling back to CPU int8 local model.` 一类的信息，说明GPU工作正常。

## 6.1 推荐模型与转写参数优化

### 推荐模型

当前工作区更推荐以下模型：

1. `fast-whisper-model/faster-whisper-large-v3-turbo`
2. `fast-whisper-model/whisper-large-v3-float16`

推荐顺序的理由：

- `faster-whisper-large-v3-turbo`
  - 速度更快
  - 显存压力更小
  - 在当前这类课程视频场景下，配合合适的转写参数后，分段和可读性已经明显改善
  - 更适合日常批量处理
- `whisper-large-v3-float16`
  - 理论上表达能力更强
  - 但显存占用更高
  - 在较长视频上更容易触发 `CUDA out of memory`
  - 如果显存有限，不建议作为默认模型

实践上更稳的默认组合是：

```text
model = fast-whisper-model/faster-whisper-large-v3-turbo
batch_size = 4
device = cuda
compute_type = float16
```

### 为什么不是只靠“换模型”

标点和分段质量不仅取决于模型，还强烈依赖转写参数和 VAD 参数。

也就是说：

- 模型决定上限
- 解码参数决定输出风格和稳定性
- VAD 参数决定分段边界

如果只换模型、不调整参数，最终结果通常仍然会出现：

- 大段连续文本
- 标点稀少
- 断句粗糙
- 术语附近的切分不自然

### 本项目参数

当前 `.agents/skills/video-study-notes/config.json` 中已经加入一组默认参数，主要包括：

- `best_of = 5`
- `patience = 1.0`
- `length_penalty = 1.0`
- `temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
- `compression_ratio_threshold = 1.4`
- `log_prob_threshold = -10.0`
- `no_speech_threshold = 0.9`
- `condition_on_previous_text = false`
- `suppress_blank = true`
- `suppress_tokens = -1`
- `without_timestamps = false`
- `word_timestamps = false`
- `prepend_punctuations = "\"'“¿([{-"`
- `append_punctuations = "\"'.。,，!！?？:：”)]}、"`
- `repetition_penalty = 1.0`
- `no_repeat_ngram_size = 0`
- `prompt_reset_on_temperature = 0.5`
- `chunk_length = 30`
- `hallucination_silence_threshold = 0.5`
- `language_detection_threshold = 0.5`
- `language_detection_segments = 1`

同时也加入了更细的 VAD 配置：

- `threshold = 0.2`
- `min_speech_duration_ms = 0`
- `min_silence_duration_ms = 2000`
- `speech_pad_ms = 400`

### 这组优化带来的实际效果

在同一个 Bilibili 视频上，迁入这组参数之后，转写结果从“长段无明显句界的大块文本”，改善成了“更短的语义片段 + 更自然的标点附着”。

例如会更接近下面这种形式：

```text
各位同学大家好,欢迎回到我强化学习的课程
那么这次是我们强化学习的第一次课
上一次课是第零次课
这次课我们将会介绍强化学习当中的基本概念
```

这仍然不等于人工精修字幕，但已经明显更适合：

- 阅读
- 做学习笔记
- 按时间段摘取重点

### 如何调整这些参数

如果你想继续试不同组合，直接改：

```text
.agents/skills/video-study-notes/config.json
```

其中最值得优先微调的是：

1. `default_model`
2. `chunk_length`
3. `vad_parameters.min_silence_duration_ms`
4. `temperature`
5. `condition_on_previous_text`
6. `batch_size`（通过命令行传入）

如果你遇到显存不足，优先做这两件事：

1. 切回 `faster-whisper-large-v3-turbo`
2. 把 `--batch-size` 降到 `4` 或 `2`

## 7. 常用脚本

### 7.1 转写音频或视频

```powershell
python ".agents\skills\video-study-notes\subskills\media-transcribe\scripts\transcribe_audio.py" --input "path\to\file.mp4" --output-dir "output\demo\transcripts" --language zh
```

输出：

- `*.txt`
- `*.srt`
- `*.transcription.json`

### 7.2 提取音频

```powershell
python ".agents\skills\video-study-notes\scripts\prepare_audio.py" --input "path\to\video.mp4" --output-dir "output\demo\audio"
```

### 7.3 提取关键帧

场景变化抽帧：

```powershell
python ".agents\skills\video-study-notes\scripts\extract_keyframes.py" --video "path\to\video.mp4" --output-dir "output\demo\keyframes"
```

按指定时间点抽帧：

```powershell
python ".agents\skills\video-study-notes\scripts\extract_keyframes.py" --video "path\to\video.mp4" --output-dir "output\demo\keyframes" --timestamps-file "output\demo\keyframe_timestamps.txt"
```

### 7.4 下载 URL 视频

```powershell
python ".agents\skills\video-study-notes\subskills\yt-dlp\scripts\run_yt_dlp.py" --no-playlist "<URL>"
```

## 8. 工作流建议

完整工作流通常是：

1. 确定输入是 URL 还是本地视频
2. 建立输出目录
3. 尝试获取字幕
4. 如果没有字幕，则提取音频并转写
5. 抽取场景帧和时间点帧
6. 生成 `notes.md`

标准输出目录结构示例：

```text
output/
  <Series Title>/
    <Video Title>/
      video/
      audio/
      subtitles/
      transcripts/
      keyframes/
      keyframe_timestamps.txt
      notes.md
```

## 9. 常见问题

### 9.1 `python` 不是 conda 环境里的解释器

检查：

```powershell
python -c "import sys; print(sys.executable)"
```

如果不是目标环境路径，说明环境没激活对。

### 9.2 `ffmpeg not found`

说明 `ffmpeg` 不在 `PATH`。安装后重新打开终端，再执行：

```powershell
ffmpeg -version
```

### 9.3 `Model path not found`

说明配置文件所指向的模型目录不存在。先检查：

```text
.agents/skills/video-study-notes/config.json
```

以及它里面的：

```text
default_model
```

是否指向一个真实存在的目录。

请确认模型目录名和位置正确。

### 9.4 `CUDA transcription failed (Library cublas64_12.dll is not found or cannot be loaded)`

这通常说明 CUDA 运行时缺失，或者 DLL 搜索路径不正确。

解决步骤：

1. 安装：

```powershell
uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12
```

2. 确认 DLL 已安装：

```powershell
Get-ChildItem -Recurse "<your_conda_env>\Lib\site-packages\nvidia" | Where-Object { $_.Name -match 'cublas64|cudnn64|cudart64|nvrtc64' } | Select-Object -ExpandProperty FullName
```

3. 使用仓库中已经修复过的转写脚本。这个脚本会自动：

- 注册 `os.add_dll_directory(...)`
- 将 NVIDIA `bin` 目录前置加入当前进程 `PATH`

### 9.5 假视频能跑 GPU，真实音频却回退到 CPU

原因不是“音频太短”或“模型有问题”，而是 Windows 下 batched CUDA 路径在真正执行 `encode()` 时对 DLL 搜索路径更敏感。

当前仓库里的 `transcribe_audio.py` 已经修复了这个问题。

### 9.6 Bilibili 没有公开字幕

有些 B 站视频在未登录时只能拿到 `danmaku`，拿不到真正字幕。这时应该：

1. 下载视频
2. 提取音频
3. 本地转写

如果你有登录态，也可以考虑在 skill 根目录放 `cookies.txt`。

### 9.7 PowerShell 里中文路径显示乱码

这通常是控制台编码显示问题，不一定代表真实文件路径错了。优先用文件搜索验证：

```powershell
python -c "from pathlib import Path; [print(p) for p in Path('output').rglob('*.mp4')]"
```

## 10. 已知说明

- 这个工作区不是完整复刻原 Linux 仓库，而是一个 Windows 精简适配版。
- 没有保留 Linux bootstrap、自动环境配置、安装器等冗余部分。
- 重点保留了：下载、转写、抽帧、笔记产出这几项核心能力。

## 11. 相关文件

- `requirements-skill.txt`
- `.agents/skills/video-study-notes/SKILL.md`
- `.agents/skills/video-study-notes/scripts/prepare_audio.py`
- `.agents/skills/video-study-notes/scripts/extract_keyframes.py`
- `.agents/skills/video-study-notes/subskills/yt-dlp/scripts/run_yt_dlp.py`
- `.agents/skills/video-study-notes/subskills/yt-dlp/scripts/resolve_project_root.py`
- `.agents/skills/video-study-notes/subskills/media-transcribe/scripts/find_local_subtitles.py`
- `.agents/skills/video-study-notes/subskills/media-transcribe/scripts/transcribe_audio.py`
