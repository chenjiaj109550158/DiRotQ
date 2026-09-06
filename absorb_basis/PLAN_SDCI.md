# PLAN_SDCI：sDCI 第二賽制，六模型 × ref/SVDQuant/ours（2026-09-06）

使用者指示：另一台 5090 跑 MJHQ-5K 全量（turbo/pixart/sana/schnell/base），
本機跑 **sDCI**，張數沿用主表協定（pixart/sana/turbo 2500、base/schnell
1000、dev 500），ref / SVDQuant / ours 三庫皆生成；AbsorbQuant 補 sDCI 支援。

## 協定（照 deepcompressor `dataset/data/DCI/DCI.py`）

- prompts：`mit-han-lab/svdquant-datasets/sDCI.yaml`（{name: prompt}）；
  GT：同 repo `sDCI.gz`（flat `<name>.jpg`）。
- 子集規則與 MJHQ **完全相同**：`random.Random(0).shuffle(keys)` → 前 N →
  sorted（DC 兩個 loader 程式碼同構；MJHQ 版已三重驗證）。
- 生成：eval_mjhq 同一迴圈（per-filename hash 種子、CPU generator、batch 1），
  ref 路徑已六家族位級對齊 DC；量化路徑 kernel 噪聲級等價。
- 指標：PSNR/LPIPS/SSIM vs ref、FID vs ref、FID vs sDCI GT；IR 另環境後補。

## AbsorbQuant 變更

`scripts/eval_mjhq.py`：`--benchmark {MJHQ,sDCI}`、`--tag`（`svdq`+`--ckpt`
生成基線第三庫）、`benchmark_subset()`、`sdci_gt_dir()`（GT 自動下載/解壓）、
指標檔名含 benchmark/tag。冒煙：pixart sDCI-3 三庫 OK。

## 執行（`sdci_chain.sh`，便宜先跑，可續傳）

| 模型 | n | 三庫估時 |
|---|---|---|
| sdxl-turbo | 2500 | ~50 分 |
| pixart | 2500 | ~3.3h |
| sana | 2500 | ~2.5h |
| sdxl-base | 1000 | ~5.3h |
| flux-schnell | 1000 | ~3h（ref bf16 offload 佔大宗） |
| flux-dev | 500 | ~7.4h（ref 50 步 offload） |

合計 ~22h GPU。產物：`workdir/<m>/eval/{ref,quant,svdq}/sDCI-<n>`；
指標 → `results/sdci/<m>_{ours,svdq}.json`。SVDQuant 側權重：pixart/sana/
sdxl 為本機重校準 kernel 檔，flux 為官方 nunchaku 權重（與主表一致）。
