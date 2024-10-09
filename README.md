该仓库旨在分析大语言模型推理时的显存占用，实验选用 Qwen2-7B 系列模型和 Transformer （4.41.2）推理框架。


使用方法：

- 使用 docker/Dockerfile 构建 csw_llm_deploy 镜像
- 从 HuggingFace 或 ModelScope 先后下载 Qwen2-7B-Instruct、Qwen2-7B-Instruct-AWQ、Qwen2-7B-Instruct-GPTQ-Int4 权重，并放置于路径 MODEL_PATH
- 使用 bash  run.sh  MAX_NEW_TOKENS  MODEL_PATH 运行测试脚本，该脚本会先后使用上述三种 LLM 对 【北京有哪些好玩的地方？】这一问题进行回答，并强行生成 MAX_NEW_TOKENS 个 tokens，同时将推理时的显存变化记录在 output/*.log 文件中
- 使用 bash  draw.sh 运行分析和绘图脚本，该脚本会将 output/*.log 文件中记录的显存变化提取至 EXCEL 表格，并绘制显存变化曲线
