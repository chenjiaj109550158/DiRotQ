# 第三輪品質優化（快速輪）：per-channel top + S 強度 α + λ 加密

狀態：**執行中**（2026-08-31 起）。
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
