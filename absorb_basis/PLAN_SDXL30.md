# SDXL-base（30 steps, CFG 5.0）部署計畫：ours vs SVDQuant，MJHQ-2500

狀態：**執行中**（2026-08-31 起）。模型 `stabilityai/stable-diffusion-xl-base-1.0`
（fp16 variant），deepcompressor pipeline name `sdxl`（內建支援）。
上游未提供 SDXL-base 的 NVFP4 recipe/yaml —— 我們以 sdxl-turbo.yaml 為模板
自建 `configs/model/sdxl.yaml`：`num_steps: 30`、`guidance_scale: 5.0`
（diffusers SDXL 預設 CFG）、protocol `euler30-g5.0`、skip 清單與 Turbo
相同（同架構 UNet 2.6B：560 transformer linears + 34 resnet convs）。

## 嚴格協定（沿用）

兩法共用同一 qdiff-128 校準 prompts。SVDQuant 用 deepcompressor 原版
pipeline（其 `quant.calib.num_samples=128` 對 prompt×step×guidance 的
cache 檔做 seeded 抽樣 —— 30 步 CFG 產生 7680 檔、抽 128 檔，活化量與
Turbo 完全相同）；ours 的 cov/λ/S 選擇只用同一批 qdiff-128 caches 與
qdiff-128 生成排名（Algorithm 1，四判準守門）。部署兩側同 nunchaku
SVDQW4A4Linear kernel、NVFP4 兩層 scale、rank 32；conv 同 Turbo 走
fused fp16（nunchaku SDXL 語意）。

## Algorithm 1 選單（round-3 定版）

λ ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}（六點排名 → λ*）；
S 守門於 λ*（SVDQuant smooth.pt + 增益 τ=+0.3dB 層選擇）×
α ∈ {0.25, 0.5, 0.75, 1.0}（貪婪逐點，≥3/4 才接受）。
per-channel top 不進選單（round-3 三模型一致負結果）。

## 執行步驟與預估時間（RTX 5090，以 Turbo 實測換算；gen 步驟 ×15 前向數）

| # | 步驟 | Turbo 實測 | SDXL-base 預估 |
|---|---|---|---|
| 0 | yaml + 腳本參數化 + 模型下載 (~7G) | — | ~1h |
| 1 | SVDQuant NVFP4 校準（collect 7680 caches ~1.5h + smooth ~5h + lowrank ~2.2h + wgts ~0.2h）+ save-model dump | 7.25h | **8–11h** |
| 2 | svdq dump → kernel 轉換 + validate | ~0.5h | ~0.5h |
| 3 | bf16 ref qdiff-128 生成 | 1 min | ~15 min |
| 4 | ours cov 收集（linear+conv，全 7680 caches 重放） | 9 min | **2–3h** |
| 5 | Algorithm-1 選單：6λ×(build 5m + gen 13–20m) + S 4α | ~40 min | **3.5–5h** |
| 6 | 官方 MJHQ-2500 ×3（bf16 ref / ours kernel / svdq kernel，30 步 CFG） | 21 min/組 | **4–5.5h/組 ≈ 12–16.5h** |
| 7 | 五指標（PSNR/LPIPS/SSIM/FID-ref/FID-GT ×2 法） | ~20 min | ~1h |
| 8 | 文件 + commit/push + vault 備份 | — | ~0.5h |

**總計 ≈ 29–39h 連續 GPU（約 1.2–1.6 天）。**
主要不確定度：step 1 的 smoothing gridsearch（活化量同 Turbo 應近 5h，
但 30 步 cache 的 batch 組成不同）；step 6 的每張 30 步×CFG 生成速度
（依 Turbo MJHQ 2.5 it/s @4 步換算 ≈ 6.3s/img）。

## 產物路徑

- yaml：`~/deepcompressor/examples/diffusion/configs/model/sdxl.yaml`
- SVDQuant dump：`models/sdxl-base/svdq_model_dump/`
- cov：`models/sdxl-base/basis/absorb_cov_sdxl_{linear,conv}.pt`
- 我們 kernels：`models/sdxl-base/absorb_basis/sdxlb_*.pt`
- 選擇/官方結果：`results/sdxlb_lambda_qdiff128.json`、
  `results/sdxlb_S_qdiff128.json`、`results/sdxlb_final_test2500.json`
- chains：`absorb_basis/sdxl/run_sdxl30_chain1.sh`（step 1–2）、
  `run_sdxl30_chain2.sh`（step 3–7）；log 在 `results/sdxlb_chain{1,2}.log`

## 防護

setsid 脫離、磁碟 guard（<50G 中止）、里程碑 vault 備份（calib 後、
選單後、官方後）、push 失敗記錄不擋跑。磁碟增量預估 ~60G
（caches 8G + dump ~5G + kernels ~25G + 圖 ~15G），現餘 206G 足夠。
