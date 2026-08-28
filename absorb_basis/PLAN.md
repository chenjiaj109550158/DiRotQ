# 計畫：absorb + H-SVD + down-absorb 之後的品質優化

> **狀態更新（2026-08-28）：A、B 已在 PixArt-Σ 上執行完畢，結果見文末。**
> B（damping 掃描）勝出：λ=0.1 三項全贏 baseline 與 SVDQuant；
> A（sequential 校準）與 baseline 打平。FLUX 端的 λ 掃描移植進行中。

基準（目前最佳，commit `0894ed7`）：`--basis hsvd --down-absorb`，MJHQ-1000 上
PSNR 18.91 / LPIPS 0.2278 / SSIM 0.7456 / FID-vs-ref 27.81 / FID-vs-GT 59.73，
全面優於 SVDQuant NVFP4（18.82 / 0.2319 / 0.7430 / 28.27 / 60.40）。

已關閉的路線（有完整負結果，不要重試）：smoothing 全家族（smooth-then-PCA
a05/svdq、解耦 main-only a05、per-layer α 搜尋）、per-group clip 搜尋、
H-metric lora refit + 交替 GPTQ、校準均值 bias correction。教訓：局部指標
（weight QSNR、per-layer 輸出 MSE）一再無法預測端到端品質，任何改動都必須
跑到 MJHQ-32 端到端才算數。

---

## B. hsvd damping 掃描（先做：幾乎免費）

**動機**：`hsvd_basis()` 的 damping（`H + λ·mean(diag(H))·I`）目前隨手設
λ=0.01。λ 越小，Cholesky 白化越徹底、basis 越激進地對準高能量 activation
方向；λ 越大越退化向 plain weight SVD。這是我方方法特有的旋鈕，從未調過。

**做法**：
1. `build_checkpoint.py` 的 `hsvd_basis(damping=...)` 加 CLI 參數
   `--hsvd-damping`（預設 0.01 保持向後相容）。
2. 掃 λ ∈ {0.003, 0.03, 0.1}（0.01 = 現有基準，不用重跑）：
   ```bash
   python absorb_basis/build_checkpoint.py --basis hsvd --down-absorb \
       --hsvd-damping 0.003 --out models/flux-schnell/absorb_basis/...damp003....safetensors
   ```
   每個 variant：build（~11 分鐘）→ validate_kernel → MJHQ-32 生成 + metrics
   （用 scratchpad/vault 備份裡的 `run_nvfp4_nunchaku.py` + `compute_metrics.py`，
   協定同前：bf16 reference 已存在，不用重生）。
3. 若某個 λ 在 MJHQ-32 全指標一致優於 0.01，再跑 MJHQ-1000 確認。

**預期**：±0.05dB 級的小增益；成本每點 ~15 分鐘。

## A. 誤差傳播式（sequential）校準（主力項目）

**動機**：目前所有 H（GPTQ Hessian + H-SVD basis + down covariance）都收自
**全精度**模型的 activation；部署時第 i 個 block 的輸入來自前 i−1 個已量化
block，分布已漂移。逐 block 用「已量化前綴」的 activation 重新校準，GPTQ 會
自動補償上游量化誤差 —— 正對「誤差沿 4 步 × 57 block 累積」的問題。

**做法**（新腳本 `absorb_basis/build_sequential.py`，全程 torch 模擬、不需
nunchaku）：
1. 載入 512 個校準 cache，先算出 embedding 後的初始 (img_hidden, txt_hidden,
   temb, rope) —— 用 bf16 FluxTransformer2DModel 的前段（x_embedder /
   context_embedder / time_text_embed 都不量化，直接用）。
2. 依序處理 19 個 double block、再 38 個 single block。對每個 block：
   a. 用**當前**（前綴已量化）的 activation 流，hook 該 block 的
      qkv/out/fc1/fc2 各輸入，累積 H（3072 維在 GPU、12288 維同
      collect_cov_down 的方式）。
   b. 對該 block 的每層：`hsvd_basis(W, H)` → `W_res = W − lora_up·D` →
      GPTQ（two-level NVFP4 grid，沿用 `quantize_residual`）→ 得
      `W_hat = W_q + lora_up·D`。
   c. 把該 block 的線性層權重**就地換成 W_hat（bf16）**（模擬量化後的權
      重；activation 量化不模擬——kernel 的 fp4 act 誤差 ~7% 是無偏的，
      模擬它成本高且 GPTQ 框架本來就只補權重側；如需更忠實可後續加
      `simulate_act_fp4`，先不做）。
   d. 打包該層的 nunchaku 張量（沿用 `pack_layer`），存進 checkpoint dict。
   e. 前傳整批 activation 通過已量化 block，進入下一個 block。
3. adaLN norm 線性層 / 未量化層照舊複製官方 tensors；single block 的
   proj_out 拆半（attn 半 / MLP 半）與現有 layer_table 一致。
4. 記憶體預算：block 邊界 activation 全批常駐 CPU（512 × (4096+256) × 3072
   bf16 ≈ 13.7 GB，54 GB RAM 內），分批搬 GPU 前傳；transformer 權重逐
   block 載入即可（不必整模型上 GPU）。
5. 輸出與現有格式完全相同的 safetensors → `validate_kernel.py` →
   MJHQ-32 → 若贏再 MJHQ-1000。

**變因控制**：先做「sequential + hsvd（λ 用 B 的最佳值）+ down-absorb」單一
variant 與現行最佳直接對比；不要同時改多個東西。

**估計**：實作約半天；執行約 1–2 小時（57 個 block × (校準前傳 + 量化)）。

**注意**：deepcompressor 的 SVDQuant 校準同樣用 fp 模型 activation，所以此
項是「兩邊都能用」的校準協定強化；比較對象是他們已發布的權重，加上去合理，
但論文敘事中應如實標註（同 H-SVD 的處理方式）。

---

## 執行順序

1. B：damping 掃描（~1 小時內出結果）。
2. A：sequential 校準，basis 用 B 的最佳 λ。
3. 兩者的贏家跑 MJHQ-1000 五指標確認（bf16 reference 與 GT 已備份於
   /vault/dirotq-absorb-backup，重跑只需生成量化側 + metrics）。

---

## 結果（2026-08-28，PixArt-Σ MJHQ-500，fake-quant 模擬，vs fp16 ref）

執行順序與原計畫相反：因當時正在做 PixArt 移植，A、B 先套在 PixArt 上
（`build_pixart_sim.py --hsvd-damping`、`build_pixart_sequential.py`、
`run_plan_ab.sh`；數據在 `results/pixart_plan_ab_mjhq500.json`）。

| 配置 | PSNR ↑ | LPIPS ↓ | SSIM ↑ |
|---|---|---|---|
| SVDQuant NVFP4 | 17.78 | 0.2929 | 0.6750 |
| baseline（λ=0.01） | 17.76 | 0.2950 | 0.6732 |
| λ=0.003 | 17.86 | 0.2923 | 0.6742 |
| λ=0.03 | 17.71 | 0.2961 | 0.6710 |
| **λ=0.1（勝者）** | **17.95** | **0.2908** | **0.6803** |
| A：sequential（λ=0.01） | 17.82 | 0.2949 | 0.6730 |

- **B 有效且翻盤**：λ=0.1 三項全面贏 baseline 與 SVDQuant —— PixArt 在
  2500 張正式比較原本微幅落後（4/5 項），λ=0.1 在同基準 500 子集轉為全勝。
  但 λ 的 landscape 非單調（0.03 反而最差），是個噪的旋鈕，跨模型不可假設
  可遷移，須各自掃描。
- **A 打平**：sequential 校準（memmap 串流版，5120 caches 全量，97 分鐘）
  PSNR 略贏、SSIM 略輸 baseline，weight-QSNR 18.86 dB 最高但端到端無感 ——
  再次印證局部指標不可信。成本高（~1.6h/次）收益零，暫時關閉；若之後
  與 λ=0.1 組合想重試，改動單一變因即可（`build_pixart_sequential.py
  --hsvd-damping 0.1`）。
- 待辦：λ=0.1 上 PixArt MJHQ-2500 對 SVDQuant 正式確認；FLUX 端
  λ∈{0.1, 0.003} 真量化（nunchaku kernel）MJHQ-500 排名進行中
  （`run_flux_damping.sh`）。
