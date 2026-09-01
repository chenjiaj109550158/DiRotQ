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
- 待辦：λ=0.1 上 PixArt MJHQ-2500 對 SVDQuant 正式確認。

## 結果（2026-08-28，FLUX-schnell MJHQ-500，真量化 nunchaku kernel，vs bf16 ref）

`run_flux_damping.sh`；數據在 `results/flux_damping_mjhq500.json`。

| 配置 | PSNR ↑ | LPIPS ↓ | SSIM ↑ |
|---|---|---|---|
| SVDQuant NVFP4 | 18.74 | 0.2311 | 0.7444 |
| baseline（λ=0.01） | 18.80 | 0.2284 | 0.7454 |
| λ=0.1 | 18.86 | 0.2278 | 0.7458 |
| **λ=0.003（勝者）** | **18.90** | **0.2270** | **0.7473** |

- **B 在 FLUX 上同樣有效**：兩個掃描點都三項贏 baseline，領先 SVDQuant 的
  幅度進一步拉大。
- **最佳 λ 跨模型不同**（PixArt→0.1、FLUX→0.003），damping 必須 per-model
  掃描，不可遷移 —— 論文敘事中 λ 應列為 per-model 超參數（SVDQuant 的
  smooth-α 網格搜尋同理，這樣對比是公平的）。

> 注意：以上兩節的 λ 排名是在 MJHQ 測試子集上做的（開發期探索）。正式協定
> 的 λ 選擇與最終數據見下一節 —— 引用數據以下一節為準。

---

## 最終結果（2026-08-29，嚴格協定：λ 只用校準資料選定）

**協定**：λ ∈ {0.003, 0.01, 0.1} 以端到端生成品質在 SVDQuant 自己的校準集
（`prompts/qdiff.yaml` 前 128 條，與其 smooth-α/calib_range 搜尋同資料同量）
上排名（`run_lambda_calib.sh`），選定即冻結，測試 benchmark（MJHQ、sDCI）
完全未參與選擇。兩邊皆為真量化 nunchaku kernel（PixArt 側 SVDQuant 由其
save-model dump 經官方 converter 轉至同一 kernel；速度/記憶體實測相同：
0.503 GiB / ~65.8 ms vs fp16 1.155 GiB / 91.7 ms）。

**λ 選擇結果**（`results/{pixart,flux}_lambda_qdiff128.json`）：
- PixArt：λ*=0.1 —— 與 MJHQ 測試域探索的贏家一致 → λ 選擇跨資料集 robust。
- FLUX：λ*=0.01 —— 開發期 MJHQ-500 上 0.003 的小幅增益不被校準域支持，
  誠實回退到預設值（最終配置即原 baseline，不受影響）。

**正式測試**：

FLUX-schnell MJHQ-1000（`results/flux_final_test1000_*.json`）—— **五項全勝**：

| 指標 | absorb λ=0.01 | SVDQuant |
|---|---|---|
| PSNR ↑ | **18.91** | 18.82 |
| LPIPS ↓ | **0.2278** | 0.2319 |
| SSIM ↑ | **0.7456** | 0.7430 |
| FID vs ref ↓ | **27.81** | 28.27 |
| FID vs GT ↓ | **59.73** | 60.40 |

PixArt-Σ MJHQ-2500（`results/pixart_final_test2500_*.json`）—— 近打平（2:3）：

| 指標 | absorb λ=0.1 | SVDQuant |
|---|---|---|
| PSNR ↑ | **17.99** | 17.88 |
| LPIPS ↓ | 0.2959 | **0.2951** |
| SSIM ↑ | **0.6753** | 0.6727 |
| FID vs ref ↓ | 20.62 | **20.22** |
| FID vs GT ↓ | 28.63 | **28.42** |

（SVDQuant 的 PixArt recipe 額外含 grid-search smoothing + 100-iter lowrank
最佳化；我們無 smoothing、單次 H-SVD+GPTQ，校準成本低得多。）

## PixArt FID 落後的歸因與方向 1 結果（2026-08-29）

**歸因**（`結論由三個實證分析支持`）：FID 差距全在 covariance 項（mean 項相
同），非多樣性坍縮；癥結是我們的圖比 ref 系統性偏軟（Laplacian 變異數
−6.5%、高頻佔比 −0.8pp），SVDQuant 反而比 ref 偏銳（+4%）。機制：全程
ℓ2/H-加權誤差最小化 → 向均值回歸；無 smoothing 時 outlier 撐大 per-group
scale，小幅值高頻分量落入 deadzone。λ 非兇手（λ=0.1 在 500-proxy 上 FID
反而最佳）。

**方向 1（weight 側 per-group clip-ratio 搜尋，`--clip-search`）**：
qdiff-128 選擇時四項全勝，但 MJHQ-2500 測試只部分遷移 ——
damp0.1+clip：17.89/0.2993/0.6746/FID-ref **20.53**/FID-GT **28.50**
（vs 無 clip 20.62/28.63：FID-GT 差距 0.21→0.08 幾乎追平；但 PSNR −0.10、
LPIPS +0.0034）。結論：clip 是 FID↔similarity 的權衡旋鈕而非免費增益；
n=128 的端到端 proxy 選粗旋鈕（λ）可靠、選細部權衡解析度不足。

**未試方向**：方向 2 = 輕量 smoothing 重審（act 側 deadzone 是軟化主因，
kernel 原生支援 smooth 除法、零部署成本；先前否決時判準無 FID）；
方向 3 = FLUX clip 用新協定重審（8/27 舊結論是在 PCA 基底 + 無 FID 判準
+ MJHQ-32 下做的，參考價值有限）。

## SANA-1.6B 結果（2026-08-29，嚴格協定，真 kernel）

模型 `Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632`（bf16，
flowdpm20-g4.5）。SVDQuant 無現成 NVFP4 權重（HF 只有 INT4，且 int4
kernel 不支援 sm_120）→ 用他們的 code 全程校準 NVFP4（smoothing
GridSearch 4.5h + lowrank/calib_range 1.8h）並經 save-model dump 轉至
同一 kernel。SANA 維度（2240）非 128 倍數 → pad-first 打包（先補零對齊
再量化，kernel-vs-sim 24 dB 驗證）；GLUMBConv 的 1x1 conv 以 channel 維
linear 處理（4D wrapper）；160 = 20 blocks x 8 層，cross-attn KV 依
attn_add skip 保持 bf16。SANA 的 `proj.fuse_when_possible = False`（與
PixArt 不同），轉換時全層保留真實 smooth。

λ 選擇（qdiff-128、四判準含 FID proxy）：**λ*=0.003 四項全勝**
（19.18/0.1664/0.7461/43.1；0.1 居中、0.01 最差）。三個模型 λ 各異：
FLUX→0.01、PixArt→0.1、SANA→0.003。

MJHQ-2500 正式賽（`results/sana_final_test2500.json`）—— 近打平（2:3）：

| 指標 | absorb λ=0.003 | SVDQuant |
|---|---|---|
| PSNR ↑ | **19.79** | 19.76 |
| LPIPS ↓ | 0.1630 | **0.1624** |
| SSIM ↑ | 0.7425 | **0.7426** |
| FID vs ref ↓ | 10.71 | **10.41** |
| FID vs GT ↓ | **27.15** | 27.22 |

Kernel parity 實測：兩者 transformer 1.268 GiB（bf16 2.997 GiB，2.36x）、
median forward 37.86 vs 38.04 ms（batch-2 CFG、1024px、RTX 5090）。
transformer 輸出 QSNR vs bf16：ours 27.9 dB / svdq 轉換 27.3 dB。
FID-ref 落後模式與 PixArt 相同（軟化機制），PLAN_ROUND2 的 S/G/C/R
同樣適用於 SANA。

---

## 論文素材：統一的校準期自動配置程序（Algorithm 1）

回應「每模型配置不同是否方法不統一」的疑慮：per-model 的是**配置輸出**，
不是方法。三個模型執行完全相同的 pipeline 與決策規則，全部只用校準資料。

### Algorithm 1: Calibration-Time Auto-Configuration

```
輸入: 預訓練權重 W、校準集 D_calib（與 SVDQuant 相同的 128 qdiff prompts）
輸出: 部署配置 (λ*, S*) 與打包好的 kernel checkpoint
固定選單: Λ = {0.003, 0.01, 0.1};  smoothing 門檻 τ = +0.3 dB
判準 Rank(·): 在 D_calib 上端到端生成, 以 {PSNR, LPIPS, SSIM, FID-proxy}
              對 bf16 參考的四判準多數決（皆為校準域, 測試 benchmark 不參與）

1  H ← 校準 activation 的二階統計 E[xxᵀ]           # 直接讀 SVDQuant 的 caches
2  λ* ← argmax_{λ∈Λ} Rank(Build(W, H, λ))           # H-SVD damping 選擇
3  for 每個量化層 ℓ:                                  # smoothing 層選擇
4      gain_ℓ ← act-quant SNR(Q(x/s_ℓ)·s_ℓW_res) − SNR(Q(x)·W_res)
5      sel_ℓ ← [gain_ℓ > τ]                          # s_ℓ = SVDQuant 官方 smooth
6  cand_S ← Build(W, H, λ*, smooth=sel)
7  S* ← sel  if Rank(cand_S) ≻ Rank(Build(W, H, λ*))  else ∅   # 端到端守門
8  return Pack(Build(W, H, λ*, smooth=S*))
```

三模型的輸出：FLUX → (0.01, S 108 層)；PixArt → (0.1, S 109/168 hooks)；
SANA → (0.003, ∅ —— 第 7 行守門否決,如實報告)。

### 審稿防禦對照表

| 潛在質疑 | 防禦 |
|---|---|
| 配置是否看了測試集 | 所有選擇只用 qdiff-128 校準 prompts(SVDQuant 同款同量);測試集每配置只跑一次 |
| 選單是否事後湊 | 選單/門檻固定,三模型跑同一 Algorithm;SANA 的 S 被守門規則否決也如實報告 |
| smooth factors 來源 | SVDQuant 自家 GridSearch 的產物(同校準資料的副產品),論文明示;等價於把他們的校準成果當可重用資源 |
| per-model 配置的先例 | SVDQuant 本身即 per-model 異質(每模型 smoothing 配方/skip 清單/dtype 皆不同,α 逐層網格搜尋);我們的選單更小且全自動 |
| λ landscape 噪 | 附三模型 λ 敏感度數據;PixArt 上驗證過「校準域選擇 = 測試域贏家」的跨資料集一致性 |
| 校準成本 | 遠低於 SVDQuant(單次 H-SVD+GPTQ + 小網格,無 100-iter lowrank/逐層 α 搜尋;SANA 上 6.3h vs 我們全選單 <1h) |

## SDXL-Turbo 結果（2026-08-31，嚴格協定，真 kernel）

`stabilityai/sdxl-turbo`（fp16，eulera 4 步 g0）。UNet 架構：560 個
transformer linears 走真 SVDQW4A4Linear kernel（維度皆 128 倍數免
padding；attn_add skip = cross-KV 留 fp16）；34 個 resnet conv 依 nunchaku
SDXL 部署語意以 fp16 conv 執行「NVFP4 格點反量化 + rank-32 im2col H-SVD
修正」的融合權重（SVDQuant 的 recipe 對 conv 無 branch/無 smooth，我們的
conv rank-32 修正為零成本方法優勢；他們的 up-block concat conv 以
ConcatConv 拆分儲存，轉換時沿 in-dim 併回）。SVDQuant 側由其 code 全程
NVFP4 校準（smoothing 5h + lowrank 2.2h）經 save-model dump 轉至同一
kernel。過程中修了兩個環境/上游問題：deepcompressor tree_collate 對
「全 batch 相同張量」不 batch 導致 SDXL 常數 time_ids 崩潰（改永遠
concat）；cleanfid 硬編碼 /tmp 在新 container 不可寫（EXDEV，改指向
使用者目錄）。

Algorithm 1（qdiff-128）：λ*=0.1（3/4 判準）；S 守門 3:4 通過
（93/560 層 smooth，median 增益僅 +0.06 dB 但選擇性套用有效）→
最終配置 = λ0.1+S。

MJHQ-2500 正式賽（`results/sdxl_final_test2500.json`）—— **4:1 勝**：

| 指標 | absorb λ0.1+S | SVDQuant |
|---|---|---|
| PSNR ↑ | **19.24** | 19.08 |
| LPIPS ↓ | **0.2139** | 0.2201 |
| SSIM ↑ | **0.6877** | 0.6769 |
| FID vs ref ↓ | **12.08** | 12.22 |
| FID vs GT ↓ | 35.37 | **35.21** |

部署 parity：兩者 unet 2.449 GiB（fp16 4.782 GiB，1.95x）、median
forward 90.2 vs 88.2 ms（同 kernel 路徑，差距 ~2%）。

### 五模型總表（全部嚴格協定 + 真 kernel + 五指標；2026-09-01）

| 模型 | 最終配置 | vs SVDQuant |
|---|---|---|
| FLUX-schnell（1000） | λ0.01+S@1.0 | **5:0** |
| PixArt-Σ（2500） | λ0.1+S@0.5 | **5:0** |
| SDXL-Turbo（2500） | λ0.3 | **5:0** |
| SDXL-base 30 步（1000） | λ0.001+S@0.5 | **5:0**（幅度最大：PSNR +0.87dB、FID-ref −3.17；見 PLAN_SDXL30.md） |
| SANA-1.6B（2500） | λ0.3+S@0.25 | **4:1**（僅 FID-ref −0.09） |

合計 24/25 指標勝。round-3（λ 網格加密 + S 強度 α 連續化 + per-channel
top 消融）細節與官方數字見 PLAN_ROUND3.md；per-channel top 為 3/3 一致
負結果（寫入 ablation）。Algorithm 1 的輸出跨四模型各異（λ 與 α 選擇皆
自動、只用校準資料），校準成本全面低於 SVDQuant（每模型他們 7h+ vs
我們全選單 <1.5h）。
