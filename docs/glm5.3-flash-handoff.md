# GLM-5.3 Flash 适配交接（2026-11-04）

本文交接当前候选源码、启动入口、已有证据与未完成事项。它不是生产验收报告。

## 交付版本与来源

- Plugin：HYGON-AI/sglang-plugin-das，`feat/glm5.3-flash`；本地基底 `405b8dc`。
- Adapter：maxiaobetter/adapter-das（HYGON-AI/adapter-das 的 fork），`feat/glm5.3-flash`；本地基底 `b8af034`。
- 两仓分别提交，必须配套使用；原开发工作树和远端实验目录未修改。
- Plugin 运行时完整同步远端 `sources/short-topk-alignment-20261104/decode/plugin`，包含其依赖，而非仅复制 GLM 文件。它与 prefill 快照仅 `models/glm5_next.py` 不同：decode 版同时允许 W8A8 和 SlimQuant 两个量化类，W8A8 prefill 条件仍保留。
- Adapter 运行时完整同步 `sources/kda-beta-fp32-candidate-20261103/adapter`。这把 FP32 beta 修正统一带入配套版本；原实验曾只给 P 使用该候选，统一版本的 D 路径仍须回归。
- Plugin 两个文件仅清除末尾多余空行以通过 diff 检查，原始/交付哈希均记录。
运行时代码及打包入口逐文件 SHA256 见 `docs/glm5.3-flash-source.json`。文档保留本地历史记录，不能把历史分数视为当前快照的成绩。
- 远端原包含通用框架兼容和其他模型依赖，因此本分支是可追溯候选快照，不是已经缩减到最小的 GLM 专属补丁。后续按依赖关系拆分评审。

## 环境与启动

用户指定后续 plugin/adapter 暂用 Python 3.12 镜像：

```text
10.16.1.152:5000/jenkins/model_test_env/vllm:0.28.1rc1-ubuntu22.04-dtk2610-py3.12-20261103-1558
```

外网仓库地址为 `42.228.13.241:5000`。21～24 已导入，镜像 ID 为
`sha256:bba1aa236bcad9dbfcbb2438c4d1d53011f6671bda634720d8bf809b95adfd33`。
这是环境物料准备完成，尚未在该镜像上完成本分支的 GLM 模型启动与精度验证。
镜像名含 vllm 不表示本次切换推理框架；本次代码仍为 SGLang 适配。

Plugin 提供 `examples/glm5.3-flash/launch.sh`、`adapter-config.json`、`deepep.json`。
脚本从远端最后候选启动参数整理：P=TP8/CP8/EP8、INT8、normal DeepEP；
D=两节点 TP16/DP16/EP16、INT4、low_latency DeepEP；EAGLE 5 steps/6 draft tokens。
节点编号 22/23/24 是脚本角色选择值，IP 由变量提供；21 不在该 P/D 拓扑内。
已移除诊断 observer 环境，脚本不启动容器、不停止其他服务，也不安装依赖。

容器内使用对应 Python 3.12 解释器，按 cookbook 安装匹配的官方 SGLang 源码与依赖；
再用 `python3 -m pip install --no-deps -e "$PLUGIN_DIR"` 注册 plugin entry point。
显式设置 `ADAPTER_DIR`、`SGLANG_DIR`、`PLUGIN_DIR`，避免旧 `/work/site` 包混入。
上游源码必须支持本候选所引用的 HookRegistry/GLM5/DSA 接口；仅 `sglang==0.5.22`
版本号不能证明兼容，原实验上游路径为 `sources/paired-layer-boundary-20261103-v2/migration`，
仍需核实并固定其来源 commit/patch 清单，不能直接换成任意最新上游。

```bash
export PLUGIN_DIR=/path/to/sglang-plugin-das
export ADAPTER_DIR=/path/to/adapter-das
export SGLANG_DIR=/path/to/compatible-sglang
export MODEL_DIR=/path/to/models
export WORK_DIR=/path/to/writable-work
export GLM53_NODE=22
export HOST_IP=<prefill-ip>
export DECODE_MASTER_IP=<decode-rank0-ip>
bash "$PLUGIN_DIR/examples/glm5.3-flash/launch.sh"
```

D 两节点分别设 `GLM53_NODE=23/24` 与本机 `HOST_IP`，其余配对变量保持一致。
模型子目录分别为 `GLM-5.3-Flash-W8A8-INT8` 和 `GLM-5.3-Flash-W4A8-INT4`。
NIC 默认 bond1/shca_0..3，须按节点真实设备核对。router/客户端仍沿 cookbook 单独启动。
脚本中的 1M context、32K chunk、HiCache/Graph 参数继承实验配置，不表示已在新镜像验收。
脚本通过 bash 语法检查；没有在本次整理期间启动模型。

必要数值开关：两角色 `SGLANG_USE_FUSED_SILU_MUL_QUANT=1`；D 的
`W8A8_SUPPORT_METHODS=3`；`SGLANG_HCU_NATIVE_MHC=1`；LightOp MLA gather 开启。
`HCU_CONFIG` 选择 DTK sparse MLA、LightOp INT8 MM、Triton grouped topk。
网页 `tools/hcu_config_ui.html` 主要管理 adapter 配置，不能据此声称可生成完整 P/D 启动脚本；
plugin `architecture.html` 是文档导览。本次以仓内 `.sh` 为启动交接入口。

## 已完成的代码与证据边界

| 环节 | 当前实现/证据 | 边界 |
|---|---|---|
| KDA prefix state | adapter 贯穿 track_state/track_chunk_idx，保存 FP32 快照；历史 cold/hit/host reload 定向验证通过 | 不等于所有长度/批次/新镜像已验证 |
| KDA beta | raw beta 在 FP32 做 sigmoid，避免转回低精度 | 已同步；与前述状态快照是两项不同修正 |
| FFN 量化及截断 | plugin GLM MLP → row_fusion → LightOp，传 swiglu_limit；W8A8/SlimQuant 接收预量化输入 | 不只加启动开关；需要模型 forward 和量化接口共同接通 |
| 两次 DeepGEMM 中间激活 | INT8 runner 保留 SiLU、clamp、量化顺序及相应分支 | normal/masked、INT4/INT8、空批次须分别核实，不能跨分支外推 |
| 短序列 KPool | 原生有序 arange 替代零 logits 的 topk 顺序；保留 PAGED/RAGGED、逻辑索引及 tail padding | 修复后同前缀复测、无插桩 badcase 仍待完成 |
| MLA / HiCache / LayerSplit / MTP | 含 native MLA、按页广播、非 owner scratch、draft host 层映射、speculative 兼容 | 历史定向探针已记录；本次整合快照未做新的 GPU 验收 |

短索引根因证据：三题、每题两次、八个 prefill rank，共 24 对；第 0～2 层完整边界及
第 3 层投影/归一化/Q 相同，首差异在相同零 logits 的 topk 索引顺序。
48 次捕获均实际命中 DTK sparse MLA；固定 Q/KV、交换索引顺序可逐位复现 24 对输出。
这证明这些输入上的 MLA 顺序因果，不能单独解释全部 GPQA 错题和复读。

## 未完成与已知问题

1. **当前精度与复读未闭环（最高优先级）**：短索引修复后的三题同前缀配对、九题无插桩回归尚缺完整报告。
   历史 P-only 融合候选九题 7/9，P+D 融合候选 6/9（错 90/102/127），均正常 stop，仍有重复片段。
   这是筛选坏例、非固定种子的单次结果，不是 GPQA 总体准确率或因果 A/B。
2. **历史全集不可继承**：较早冻结配置 HE 97.56%、MATH500 95.40%、GPQA 90.40%、HMMT 80.00%；
   对照 97.56/95.60/89.90/80.00。后续代码和环境已改变，最终候选四项全集必须重跑，保留截断与失败。
3. **镜像对齐未完成**：文档 Python 3.10 镜像尝试在 DeepEP Buffer.sync → ROCSHMEM GetClosestNicToGpu 崩溃；
   单卡可见性兼容是候选线索，未证实根因。cwd=/ 配合 PYTHONPYCACHEPREFIX 的启动失败已通过显式工作目录规避。
   Python 3.12/DTK26.10 需要重新核对 Torch、LightOp、FlashMLA、DeepGEMM、DeepEP/Mooncake 版本和共享库哈希。
4. **所有路径审计仍待完成**：长序列 pooled topk 的 AITER/LightOp dispatch、MQA 量化、KPool 写入/尾状态、
   head-gate 精度、MTP index sharing、prefill chunk/cache hit/host reload、decode eager/graph 与 verify/draft 分支。
   每项记录 caller、实际 kernel、shape/dtype、量化/截断/索引顺序及状态输入，不以启动成功代替对齐。
5. **性能未对齐**：历史热点为 ragged MQA、INT8 MM、CP 重排/广播及 expert token 不均衡；
   有独立优化收益但无最终同环境端到端验收。D2H 临时方案曾退化，不作为推荐配置。
6. **运行风险**：曾因独立 GPU 探针与 resident 模型争用触发 NUMA 内存策略 OOM；复测应使用明确独占窗口。
7. **工程收口**：远端运行时包含其他模型通用依赖、历史回退及诊断入口；需继续按归属拆分，补齐正式版本矩阵。
   本次不新增单测、不启动完整测试或 GPU 评测；保留当前源码而不把候选接口重写成另一套行为。

## 接手顺序与证据位置

先固定两仓提交、官方源码、镜像和核心库哈希，检查进程实际 import 来源；再按 prefill/decode/verify
做逐分支清单、三题同 token 前缀对比、九题并发 9 的无插桩坏例及复读检查；通过后跑全集与同配置 profile/性能。
任何失败保留原请求、响应、token 上限与 finish_reason，不将重复推理仅按最终答案正确忽略。

原始远端任务根目录由维护者提供为 `GLM53_TASK_ROOT`，不要依赖临时 `/tmp` 脚本。
关键证据相对路径：

- `logs/mla-internal-repeat-20261104-v2-complete.tar.gz`，SHA256 `6d37c690b24f71e4016b7caa2cafef278afbd2c2e128d119ea18e330257fbb0b`。
- `logs/ffn-lightop-pd-gpqa-badcase-20261103-v1`：P+D 九题原始输出与 review。
- `logs/short-topk-aligned-repeat-20261104-v2`：本次只读检查时 client.log 仍记 waiting warmup、cases_sent=0，不能认定修复后验收通过。
- `scripts/short-topk-aligned-repeat-20261104-v2`：历史启动与诊断脚本。
- 两仓 `docs/glm5.3-flash-source.json`：已同步源码的来源与 SHA256；配套历史细节见 plugin 的 `docs/hcu-sglang-development.md`。

## 本次验证

两仓使用 Python 3.12 执行 `ci/l0_gate.py`；plugin 还按 runner 配方检查 guide/migration-plan 快照；
两仓执行 `git diff --check` 与暂存差异检查。Python AST 语法检查包含在 L0 中，启动脚本运行 `bash -n`。
这些是静态交付验证，不代表新镜像上的模型精度、性能或页面浏览器渲染验收。
仓库当前未绑定 GitHub Actions 的自动检查不能写成“GitHub CI 已通过”。
