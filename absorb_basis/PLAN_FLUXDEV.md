# FLUX.1-dev（50 steps, guidance 3.5）部署計畫：ours vs SVDQuant，MJHQ-1000

狀態：**執行中**（2026-09-01 起）。模型 `black-forest-labs/FLUX.1-dev`
（bf16），與 schnell 同架構（19 double + 38 single、dims 相同、官方
nunchaku checkpoint 佈局相同）→ 全部沿用 schnell 的 build/gen 工具鏈
（build_checkpoint / flux_gen_nunchaku，v1 C++ loader）。

## 基線與嚴格協定

- **SVDQuant 側零校準**：官方 `mit-han-lab/nunchaku-flux.1-dev`
  `svdq-fp4_r32-flux.1-dev.safetensors` 直接下載（作者自校準的最強基線，
  與 schnell 做法一致）；同一檔也作為我們 build 的 `--official`
  容器與 S 的 smooth 因子來源。
- ours 校準：同一 qdiff-128 prompts（models/flux-schnell/calib_prompts.yaml）
  × 50 步 × 1 guidance（dev 為蒸餾內嵌 guidance，無 CFG×2）= 6400 caches。
  cov 主表全量重放；12288 維 down cov 與 act samples 用 1/5 strided 子集
  （1280 檔，涵蓋全部 prompts × 每 5 步）。
- Algorithm-1 選單（round-3 定版）：λ∈{0.001,0.003,0.01,0.03,0.1,0.3} 六點
  qdiff-128 四判準排名 → S（官方 smooth 因子 + 增益 τ=+0.3dB 層選擇）
  × α∈{0.25,0.5,0.75,1.0} 貪婪守門。per-channel top 不進選單。
- 官方測試：MJHQ-500（2026-09-01 使用者裁定，原計畫 1000）（bf16 ref / ours / SVDQuant，皆 nunchaku C++ kernel
  路徑），五指標，GT=MJHQ-GT-500（自 GT-2500 依生成檔名對映建出）。

## S 增益量測（新腳本 measure_smooth_gain_flux.py）

schnell 的 smooth_gain.json 生成腳本未入 git；本輪補一個正式腳本，
改用離線法：act samples（collect_act_samples.py 已有）+ cov + 官方
smooth 因子，逐層算 e0=‖fp4(X)Wresᵀ−XWresᵀ‖² 與
e1=‖fp4(X/s)(Wres·s)ᵀ−XWresᵀ‖²，gain=10log₁₀(e0/e1)。與 build 的
select-smooth 語意一致（s 取自官方 checkpoint 的 packed smooth，
build 時 s^α）。免整模型 forward，~10 分鐘。

## 步驟與預估（RTX 5090；bf16 455ms/fwd、量化 159ms/fwd 實測換算）

| # | 步驟 | 預估 |
|---|---|---|
| 1 | 校準資料收集（128×50，bf16 cpu-offload） | ~1.5–2h |
| 2 | 官方 dev NVFP4 權重下載（~7G） | ~10 min |
| 3 | cov 主表（6400 全量，batch 2） | ~1h |
| 4 | cov down（12288 維，1280 檔 × 13 passes） | ~2–2.5h |
| 5 | act samples（1280 檔） | ~15 min |
| 6 | bf16 ref qdiff-128（50 步） | ~50 min |
| 7 | λ 六點：build（12B GPTQ，~30–40m）+ gen（~20m）× 6 | ~5–6h |
| 8 | rank-λ + 落選 checkpoint 清理（每檔 7.1G） | ~10 min |
| 9 | S：gain 量測 + 4α ×（build+gen） | ~3.5–4h |
| 10 | 官方 MJHQ-500：bf16 ref ~5.3h + ours ~1.3h + svdq ~1.3h | ~8h |
| 11 | 五指標 + 文件 + 備份 | ~1h |

**總計 ≈ 26–29h。** 磁碟增量峰值 ~90G（dev 模型 23.8G + caches ~20G +
λ 檔 6×7.1G + 圖），現餘 131G，rank 後清落選檔；guard <40G 中止。

## 產物路徑

- caches：`models/flux-dev/calibration_dataset/caches`（+ `caches_sub` strided）
- cov：`models/flux-dev/basis/absorb_cov_basis.pt`、`absorb_cov_down/`、
  `absorb_act_samples.pt`
- 官方權重：`models/flux-dev/svdq-fp4_r32-flux.1-dev.safetensors`
- 我們 checkpoints：`models/flux-dev/absorb_basis/fluxdev_*.safetensors`
- 結果：`results/fluxdev_lambda_qdiff128.json`、`fluxdev_S_qdiff128.json`、
  `fluxdev_final_test500.json`
- chain：`absorb_basis/run_fluxdev_chain.sh`，log `results/fluxdev_chain.log`

## 執行結果（2026-09-02，全程 Algorithm 1 嚴格協定 + nunchaku C++ kernel）

### Algorithm-1 選擇（qdiff-128）

- λ 六點排名 → **λ\*=0.3**（總積分勝 0.1；round-3 加密的網格點再次得分）。
- S 守門：逐層增益量測強（228 層 median +0.42dB、138 層過門、零負層），
  但端到端四判準 α=0.25/0.5/0.75/1.0 全部未過（1:3/1:3/2:2/2:2）
  → **最終配置 = 純 damp0.3（無 S）**。dev 與 schnell（S@1.0 過門）相反
  —— 展示了「逐層增益 ≠ 端到端增益」與守門機制的價值。
- 選擇 JSON：`fluxdev_lambda_qdiff128.json`、`fluxdev_S_qdiff128.json`。

### 官方 MJHQ-500（50 步 g3.5，`fluxdev_final_test500.json`）— **5:0 全勝**

| 指標 | ours damp0.3 | SVDQuant（官方 dev NVFP4） | 差距 |
|---|---|---|---|
| PSNR ↑ | **21.57** | 21.06 | +0.51 dB |
| LPIPS ↓ | **0.1929** | 0.2111 | −0.018 |
| SSIM ↑ | **0.8196** | 0.8047 | +0.015 |
| FID vs ref ↓ | **36.45** | 39.39 | −2.94 |
| FID vs GT ↓ | **93.96** | 94.95 | −0.99 |

### 實測時間（RTX 5090）

collect 2h04m；cov-main 2h10m + cov-down 2h38m + act-samples 11m；
bf16 qdiff-ref 1h22m；λ 選單 6×~30m + rank ≈ 3h；S 量測 3m + 4α ≈ 2h；
官方：bf16 ref 500 5h42m + ours 1h13m + svdq 1h13m + 指標 ~25m。
全程 ~22.5h。SVDQuant 側零校準（官方權重）。

### 事件

官方段執行中改 1000→500 時，執行中 bash 使用改檔前緩衝內容開跑 1000 版
→ 殺鏈、以獨立續跑腳本 `run_fluxdev_official500.sh` 完成（教訓：
執行中腳本不可改，改未達段落須停鏈或用續跑腳本）。
