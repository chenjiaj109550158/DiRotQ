# PLAN_5K：補齊論文精確 MJHQ-5K 賽制（PixArt-Σ）

狀態：執行中（2026-09-04）。背景：原文（2411.05007）賽制為 MJHQ-30K
抽 5K（deepcompressor MJHQ.py `random.Random(0).shuffle` 前綴，決定性，
已三重驗證）；我們現有 2500 圖庫 == 該 5K 的嚴格前綴子集，且
per-filename hash 種子與 N 無關 → 既有 2500 張全數可重用，只補缺的
2500 張/庫。

## 使用者裁定的順序

1. **Phase 1（本輪）**：fp16 ref + svdq-NVFP4 兩庫補到 5K → 以 DC 原生
   eval 語義算五指標 → 與論文表對數 → 回報。
2. Phase 2（待裁定）：ours-NVFP4 + ours-MX + svdq-MX。

## 論文對照目標（PixArt-Σ，MJHQ-5K）

| | FID vs GT ↓ | IR ↑ | LPIPS(vs 16bit) ↓ | PSNR(vs 16bit) ↑ |
|---|---|---|---|---|
| FP16 | **16.6** | **0.944** | — | — |
| SVDQuant NVFP4 | **16.6** | **0.940** | **0.271** | **18.5** |
| （SVDQuant INT4 參考行） | 19.2 | 0.878 | 0.323 | 17.6 |

## 協定語義（照 DC eval 原碼，非自製）

- 生成：DC pipeline factory + per-filename hash 種子；`MJHQ-5000` 目錄
  由既有 2500 硬連結預填，DC generate 逐檔跳過已存在 → 只生成缺的。
- fp16 ref：`run_pixart_sim_generate.py --sim-weights EMPTY`（空 dict =
  純 fp16；qdiff-ref 同法前例）。
- FID vs GT：`compute_fid(dataset(MJHQ,5000,return_gt), gen_dir)` —— GT
  特徵即論文的 5K GT 子集（cleanfid backbone，同庫）。
- LPIPS/PSNR/SSIM：`compute_image_similarity_metrics(ref5000, gen)`。
- IR：ImageReward-v1.0（ir_env），prompts 對映 `mjhq_5000_samples.json`。

## 步驟/估時（GPU）

| # | 步驟 | 估 |
|---|---|---|
| 1 | ref-5000 補 2500 張（dpm20 fp16 @1024） | ~70m |
| 2 | svdq-5000 補 2500 張（NVFP4 kernel） | ~70m |
| 3 | FID×2（GT 特徵 5K 一次 + 快取）+ 相似度 + IR×2 | ~30m |
| 4 | 對表回報，等 Phase 2 裁定 | — |

產物：`results/pixart_5k_phase1.json`；GT 特徵快取
`benchmarks/stats/MJHQ/MJHQ-5000.npz`。磁碟 +~8G（167G 可用）。
