# 闪烁体脉冲反卷积服务（pulse-deconvolve）

高计数率下，相邻闪烁事件的响应尾部会互相叠加，直接把局部峰值当成独立脉冲会产生误判。本服务从离散采样中**联合**反演事件序列：在所有可起始位置上联合选择非负整数幅度，用零填充离散卷积回算全部采样，而不是逐峰扣除响应尾部。

## 数学模型

观测采样 `s[0..n-1]` 建模为事件幅度序列 `x[0..n-1]`（非负整数，每项 ≤ `maxAmplitude`）与正整数响应核 `k[0..m-1]` 的零填充离散卷积的前 `n` 个输出：

```
pred[i] = Σ_{j=0}^{m-1} k[j] · x[i-j]      （t < 0 时 x[t] = 0）
residual[i] = samples[i] - prediction[i]
```

### 有限精度标定：响应核不确定性

请求可携带 `kernelTolerance`（与 `kernel` 等长的非负整数列表），此时每个核抽头可在
`[kernel[j]-tol[j], kernel[j]+tol[j]]` 内**独立**取任意整数（每个抽头的最低可能值仍须为正）。
由于各抽头与非负幅度相乘，固定事件列在每个采样点上的预测闭区间在抽头盒子的端点处取得：

```
pred_lo[i] = Σ_j (kernel[j]-tol[j]) · x[i-j]
pred_hi[i] = Σ_j (kernel[j]+tol[j]) · x[i-j]
```

恢复出的事件列必须让**每种**允许响应核的逐点残差都不超过 `maxResidual`，即要求
`samples[i] - maxResidual ≤ pred_lo[i]` 且 `pred_hi[i] ≤ samples[i] + maxResidual`。

求解目标按字典序依次最小化：

1. **最坏最大绝对残差**（所有允许核上的最大 `|residual[i]|`；硬约束：≤ `maxResidual`）
2. **逐点最坏绝对残差之和** `Σ_i max_kernel |residual[i]|`
3. **事件数**（非零幅度个数，硬约束：≤ `maxEvents`）
4. **幅度序列本身**的字典序

省略 `kernelTolerance` 时，允许核集合只有标称核一个，上述目标退化为原有的标称目标。

由于卷积矩阵是带状下三角的，固定 `x[0..p-1]` 后采样 `p` 只依赖最近 `m-1` 个幅度，且给定逐点残差半径时 `x[p]` 的可行取值（即使在核不确定性下）仍是连续整数区间。求解器据此做精确动态规划（从半径 0 起倍增定位最小可行半径，再在该半径下优化其余目标），保证全局最优，而非贪心逐峰剥离。

## API

### `POST /api/pulses/deconvolve`

请求体：

```json
{
  "samples": [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0],
  "kernel": [3, 2, 1],
  "maxResidual": 0,
  "maxAmplitude": 10,
  "maxEvents": 4
}
```

| 字段 | 约束 | 含义 |
| --- | --- | --- |
| `samples` | 12–36 个整数 | 观测采样 |
| `kernel` | 3–6 个正整数 | 标称脉冲响应核 |
| `maxResidual` | ≥ 0 的整数 | 统一逐点残差上限 |
| `maxAmplitude` | ≥ 1 的整数 | 事件整数幅度上限 |
| `maxEvents` | ≥ 0 的整数 | 最大事件数 |
| `kernelTolerance` | 可选；3–6 个非负整数，长度与 `kernel` 相同，且每个抽头 `kernel[j]-tol[j] ≥ 1` | 每抽头有限精度容差；给出后事件列必须覆盖全部允许核 |

**成功（200）**：给出事件位置与幅度、完整幅度序列、完整预测波形、逐点残差及目标值。
省略 `kernelTolerance` 时响应与以往完全一致；给出时额外返回逐点 `predictionIntervals` /
`residualIntervals`（各为 `[low, high]` 闭区间，覆盖全部允许核产生的预测与残差），而
`prediction` / `residuals` 仍是**标称核**的结果，`objectives` 中的残差目标为最坏情形。

```json
{
  "status": "ok",
  "events": [{"position": 4, "amplitude": 5}, {"position": 6, "amplitude": 4}],
  "amplitudes": [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "prediction": [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0],
  "residuals": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "objectives": {"maxAbsResidual": 0, "sumAbsResidual": 0, "eventCount": 2}
}
```

稳健请求示例（首个核抽头可在 `{2,3,4}` 中取值，逐点残差上限 5）：

```json
{
  "samples": [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0],
  "kernel": [3, 2, 1],
  "kernelTolerance": [1, 0, 0],
  "maxResidual": 5,
  "maxAmplitude": 10,
  "maxEvents": 4
}
```

成功时 `predictionIntervals[4]` 为 `[10, 20]`、`predictionIntervals[6]` 为 `[13, 21]`，
对应 `residualIntervals` 为 `[-5, 5]` 与 `[-4, 4]`，`objectives` 为
`{"maxAbsResidual": 5, "sumAbsResidual": 9, "eventCount": 2}`。

**不可反演（422）**：当事件数或任一点残差无法满足限制时（给出容差时，指不存在能让**全部**
允许核都满足限制的完整事件列），返回可识别的结论，并指出**首个无法被任何完整候选解释的采样位置**
——即不存在这样的完整幅度序列（各项在 `[0, maxAmplitude]` 内、事件数 ≤ `maxEvents`）能让
`samples[0..p]` 在每种允许核下的残差全部落在 `±maxResidual` 内的最小 `p`。

```json
{
  "status": "infeasible",
  "detail": "no event train within the amplitude and event-count limits can reproduce the samples within the residual limit",
  "firstUnexplainablePosition": 5
}
```

**请求非法（400）**：字段缺失、采样数不足/过多、响应核长度或取值越界等，返回 `{"status": "invalid_request", "detail": [...]}`。

### `GET /healthz`

健康检查端点，返回 `{"status": "ok"}`；Docker 健康检查用它确认服务可接单。

## 运行（Docker Compose）

```bash
docker compose up --build          # 启动 API，并在其健康后自动运行一次 verify
docker compose up api              # 只启动 API
API_HOST_PORT=9000 docker compose up --build   # 自定义宿主机端口（默认 8000）
```

- API 以可配置宿主机端口暴露：`API_HOST_PORT` 环境变量（默认 `8000`），容器内固定为 `8000`。
- `api` 服务配置了健康检查（轮询 `/healthz`），确认服务可接单。
- `verify` 是一次性服务：`depends_on: service_healthy` 等待 API 健康后，依次执行
  1. **代码测试**（pytest 单元/API 测试套件）；
  2. **应用构建检查**（字节码编译全部源码、导入 ASGI 应用、校验路由与 OpenAPI 模式）；
  3. **重叠脉冲 API 冒烟**（精确重叠脉冲恢复、带噪重叠脉冲、残差上限不可满足、事件数不可满足、请求校验、核容差稳健成功、稳健无解与容差字段校验）。

  完成后自行退出，退出码汇总结果：`0` 全部通过；否则为位掩码（`1` 代码测试失败，`2` 构建检查失败，`4` 冒烟失败）。

CI 中可用以下命令把 verify 的退出码作为整条命令的退出码：

```bash
docker compose up --build --exit-code-from verify verify
```

## 本地开发

```bash
pip install -r requirements-dev.txt
pytest                                   # 运行测试
uvicorn app.main:app --reload --port 8000
```

## 项目结构

```
app/
  deconvolve.py   # 联合整数反卷积求解器（动态规划，非逐峰剥离）
  schemas.py      # 请求/响应模型与校验
  main.py         # FastAPI 路由、健康检查、错误语义
tests/
  test_deconvolve.py  # 求解器测试（含暴力枚举交叉验证）
  test_api.py         # API 测试
verify/
  run_checks.py   # verify 一次性服务入口
Dockerfile          # 多阶段：base / api / verify
docker-compose.yml  # api（健康检查、可配置端口）+ verify（一次性）
```
