# 第三輪品質優化（快速輪）：per-channel top + S 強度 α + λ 加密

狀態：**已完成**（2026-08-31，結果見文末「執行結果」）。
基準：四模型最終配置（見 PLAN.md 總表）——FLUX 5:0、PixArt 5:0、
SDXL-Turbo 4:1、SANA 2:3。

共同協定不變（Algorithm 1）：所有選擇只用 qdiff-128 校準 prompts、四判準
（PSNR/LPIPS/SSIM/FID-proxy）守門；配置改變才重跑官方測試集。部署逐 bit
不變（同 kernel、rank 32）。

## 項目 1：全層 per-channel top scale（主打）

**現狀**：我們所有 "plain" 層用 per-tensor top（bf16）× per-group e4m3
micro。**查證**：SVDQuant 只在 FLUX fused-qkv 用 per-channel top
（官方 checkpoint 的 wcscales）；其 PixArt/SANA/SDXL recipes 全部
per-tensor（dump scale.0 = (1,1,1,1)、recipe 字串 `tsnr`）。

**做法**：builder 加 `--per-channel-top`——quantize_residual/pack_layer
走 kind="qkv" 路徑（per-channel top → wcscales，wtscale=1）。kernel 原生
支援（wcscales 本就配置、語意已驗證），**零 runtime 成本**。weight 格點
按輸出通道細化——打 clip 想打而打壞的目標（正交且不動 micro 分配）。

**範圍**：PixArt/SANA/SDXL（python SVDQW4A4Linear 注入路徑，完全可控）。
FLUX 暫緩——其 v1 C++ loader 對非 qkv 層是否吃 wcscales 未驗證。

## 項目 3：S 強度連續化（s^α）

**動機**：S 目前是二元的（整層套 SVDQuant 的 s 或不套）。部分強度
s^α（α<1）可用較小的 weight 側代價收割 act 側增益——對 SANA/SDXL 這種
「增益弱、代價吃掉收益」的模型可能翻正；對 PixArt 可能再擠一點。

**做法**：builder 加 `--smooth-alpha`（s_eff = s^α，H/(s_eff s_effᵀ)
解析縮放）。層選擇沿用 α=1 的 gains + τ=0.3（單變因，論文揭露）。
α 網格 {0.25, 0.5, 0.75, 1.0}。FLUX 同步支援
（build_checkpoint --select-smooth-alpha）。

## 項目 5：λ 網格加密

{0.003, 0.01, 0.1} → 加 {0.001, 0.03, 0.3} 共 6 點。landscape 非單調，
每模型可能各撿 0.05–0.1 dB。

## 已排除項

- **GPTQ act-order**：查證後我們的 GPTQ 已內建（importance argsort +
  perm/inv_perm）；SVDQuant 根本不用 GPTQ（RTN + calib_range）。無增益空間。
- **計畫 R（block 重建）**：保留為下一輪大槓桿——若本輪後 SANA/SDXL 仍未
  翻到 4:0/5:0 再上。
- rank 重分配 / cross-attn KV 量化 / C / G / sequential / 全面 smoothing /
  clip：均有負結果或裁定不做（見 PLAN.md / PLAN_ROUND2.md）。

## 執行序（每模型，貪婪逐段 + 守門）

1. **Stage A**：λ 六點排名（舊三點的 build+gen 從 vault 取回，只補三個新點）→ λ*
2. **Stage B**：λ* + per-channel-top 一個 build → 守門 vs per-tensor
3. **Stage C**：B 勝者 + S-α 網格（含 α=1 即現行 S）→ 守門
4. 最終配置若異於現行 → 官方測試集重跑 + 五指標 vs SVDQuant

模型順序：PixArt → SDXL → SANA → FLUX（FLUX 無 Stage B）。
預估：選擇輪全程 ~5h；官方重跑視改變數量 +1~6h。

## 磁碟/還原註記

本機是新 container：PixArt/SANA/FLUX 的 cov、舊 λ checkpoints、qdiff
ref/gen 圖需自 vault `cp -au` 取回（dump 的 smooth.pt 是 symlink，取回
須 `cp -Lu` 解參照）。預估取回 ~15G，磁碟餘裕充足（575G）。

## 執行結果（2026-08-31，round3_driver.py，全程 Algorithm 1 嚴格協定）

選擇 JSON：`results/{model}_round3_selection.json`；官方重跑：
`results/{model}_round3_test.json`。逐項結論：

### 項目 1（per-channel top）：**3/3 一致負結果，棄用**

PixArt、SDXL、SANA 三者的 Stage B（λ* + `--per-channel-top`）在 qdiff-128
四判準守門全數敗於 per-tensor 基底（FLUX 因 v1 C++ loader 未跑）。
解讀：兩層 scale（per-tensor top × per-group-16 e4m3 micro）中的 micro
已吸收逐行動態範圍，top 細化成 per-channel 反而稀釋 e4m3 micro 的表示
精度。作為 negative result 寫入論文 ablation（含 fp16 underflow 的
normalized-split 實作已驗證正確，非實作瑕疵——kernel-vs-sim 22–24 dB）。

### 項目 5（λ 網格加密）：SDXL、SANA 換新贏家 λ=0.3

| 模型 | λ*（六點 qdiff-128 排名） | 變化 |
|---|---|---|
| PixArt | 0.1 | 不變 |
| SDXL-Turbo | **0.3** | 原 0.1 |
| SANA | **0.3** | 原 0.003 |
| FLUX | 0.01 | 不變 |

### 項目 3（S 強度 s^α）：PixArt α=0.5、SANA α=0.25，SDXL/FLUX 不套/α=1

- PixArt（λ0.1 上）：α=0.25/0.5 皆過門，0.5 勝 0.25（3:1）、0.75 全敗、
  1.0 對 0.5 全敗 → **S@0.5**（原 S@1.0）。
- SANA（λ0.3 上）：**S@0.25** 過門（原無 S）。
- SDXL（λ0.3 上）：全部 α 未過門 → 純 λ0.3（原 λ0.1+S@1.0）。
- FLUX：S@1.0 維持（配置不變，未重跑官方）。

### 官方測試集重跑（配置有變的三模型，MJHQ-2500、五指標、真 kernel）

| 模型 | 新配置 | ours | SVDQuant | 結果 |
|---|---|---|---|---|
| PixArt-Σ | λ0.1+S@0.5 | 18.44 / 0.2752 / 0.6890 / 19.47 / 28.27 | 17.88 / 0.2951 / 0.6727 / 20.22 / 28.42 | **5:0** |
| SDXL-Turbo | λ0.3 | 19.23 / 0.2140 / 0.6869 / 12.13 / 35.01 | 19.08 / 0.2201 / 0.6769 / 12.22 / 35.21 | **5:0**（原 4:1，FID-GT 翻正） |
| SANA-1.6B | λ0.3+S@0.25 | 19.92 / 0.1592 / 0.7471 / 10.50 / 27.01 | 19.76 / 0.1624 / 0.7426 / 10.41 / 27.22 | **4:1**（原 2:3；僅 FID-ref −0.09） |

（指標序：PSNR↑ / LPIPS↓ / SSIM↑ / FID-vs-ref↓ / FID-vs-GT↓。）

註：PixArt 舊配置 S@1.0 的官方值（18.57/0.2718/0.6919/19.57/28.09）與新
配置互有增減，但 Algorithm 1 嚴格不回看測試集，qdiff-128 上 S@0.5 四項
全勝 S@1.0，故依協定定案 S@0.5——兩配置對 SVDQuant 皆 5:0，結論不受影響。

### 輪後總結

四模型戰績：FLUX **5:0**、PixArt **5:0**、SDXL-Turbo **5:0**、SANA
**4:1**（19/20 指標勝）。本輪淨效果：SDXL 4:1→5:0、SANA 2:3→4:1、
PixArt/FLUX 持平。計畫 R（block 重建）依裁定保留為下一輪槓桿
（SANA FID-ref 是唯一殘存負項）。
