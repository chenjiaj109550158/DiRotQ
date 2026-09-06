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

## 驗證（2026-09-06）

- 子集規則 vs DC `get_dataset("DCI", N)`：N=500/1000/2500 檔名**逐一同序**。
- GT：sDCI.gz 10.8 GB，解壓 11186 張（8028 條 prompt 為其子集），workdir
  `gt/sDCI` 以 symlink 指向 DC 解壓目錄（磁碟單一副本）。
- 冒煙：pixart sDCI-3 ref/ours/svdq 三庫生成 OK。鏈 07:51 起跑（turbo 首座）。

## 每張耗時實測與 ETA 修正（2026-09-06 15:40）

- schnell ours（NVFP4 kernel）：**0.851 s/張**（1000 張，forward 中位 163.7 ms×4）。
- schnell ref（bf16，cpu-offload）：**~20 s/張**（本機探針 19.98；08-27 前機
  1000 張 5h38m ≈ 20.3）。23× 差主要是 offload 串流 22 GB transformer，
  kernel 級對比為 per-forward 455 vs 164 ms ≈ 2.8×。
- 推論 dev ref（50 步 offload）≈ 250 s/張 → 500 張 ≈ 35h，原估錯 ~10×。
  修法：eval_mjhq flux ref 改「兩階段駐留」（先全 prompt 編碼→卸編碼器→
  transformer 駐留），預期 dev ~23 s/張、schnell ~1.8 s/張；以 3+3 張位級
  閘門（vs 今日 DC 原生 / stored）把關，失敗自動還原 offload 版。
- 實測生成速率（sDCI）：pixart ref 1h40m、ours 1h18m（~2.4 s/張，非原估 1.6）。
- 兩階段駐留閘門：schnell 3/3 BIT（8.6 s/張）、dev 3/3 BIT（28.6 s/張，
  峰值 24.7 GB）→ 採用。dev ref 500 張 ≈ 4h（非 35h）、schnell ref 1000 ≈ 2.4h。
  修正後全鏈 ETA：base ~22:30 → schnell ~03:30 → dev ~10:30（明早）。
