# MXFP4e2 客場輪：ours vs SVDQuant on PixArt-Σ（PLAN_MX）

狀態：**執行中**（2026-09-04 使用者裁定：選項 1（svdq 全重校準）+
真量化 kernel + 2500 張）。目的：KroQuant/LoRaQ 賽制對齊的格式下
量測 ours vs SVDQuant 幅度（回應「NVFP4 主表邊際小」＝基線飽和）。

## 格式定義（OCP MXFP4，與 KroQuant 對齊）

- 元素：e2m1（±{0,.5,1,1.5,2,3,4,6}）；block=**32**（沿輸入維）；
- 共享 scale：**E8M0 二冪**，exponent = floor(log2(absmax)) − 2
  （e2m1 emax=2），權重與活化同規；活化逐 token 動態。

## Kernel（真量化、Triton）

`MXW4A4Linear`：權重存 packed int4 碼 + 每組 int8 exponent；
forward 單 kernel：act group-32 amax→e8m0→e2m1 量化（融合）→
權重解碼（碼×2^exp）→ **fp16 tensor-core dot**（e2m1 積於 fp16
精確⇒逐位等於真 4-bit MMA 結果）→ lora fp16 GEMM 加回。
誠實註記：計算單元 fp16、速度≈fp16；Blackwell 原生 fp4 MMA 為
後續工程。驗證：vs torch 參考實作逐位對拍（隨機+真實層）。

## 公平協定

- svdq 側：deepcompressor **加 e8m0 scale dtype 補丁**後，以其原生
  pipeline（含逐層 GridSearch smoothing，objective 在 MX 誤差下）
  全重校準（config: 仿 nvfp4.yaml 改 group 32 + e8m0）→
  dump → 轉 MX kernel .pt。
- ours 側：cov/act 統計沿用（格式無關）；λ 網格、S 增益（MX act
  sim 重測）、α 守門全部重選；GPTQ 於 MX 網格（group-32、
  scales_override）。
- 兩邊格式實作**位級對拍**（我方 quantizer vs 補丁後 deepcompressor）。
- 同 caches、同 seeds、同 ref-2500/GT-2500（磁碟現存）。

## 步驟與估時

| # | 步驟 | 估 |
|---|---|---|
| 1 | MX 參考 quantizer（torch）+ OCP 單元驗證 | 2h |
| 2 | GPTQ/builder MX 模式（group-32 scales_override）+ MX kernel .pt 打包格式 | 3h |
| 3 | Triton MXW4A4Linear + 逐位對拍 + runner 接線 | 1 天 |
| 4 | deepcompressor e8m0 補丁 + 位級對拍 + svdq MX 配置 | 3h |
| 5 | svdq 重校準鏈（GPU） | 5–7h |
| 6 | ours 選單：λ 6 點 + S×α（kernel gen ~5-6 分/config） | ~2.5h |
| 7 | finals 2500×2（kernel）+ 五指標 | ~2h |
| 8 | 記錄 + 備份 | 0.5h |

合計：工程 ~2–2.5 天、GPU ~10–12h。產物：
`results/pixart_mx_{qdiff128,test2500}.json`、`PLAN_MX.md` 結果節。

## 執行結果（2026-09-04 08:20 收官；全輪 GPU ~13h + 工程 ~1 天）

### 校準時間（同機同資料，實測）

- SVDQuant MX 全重校準：**3h28m → 1 配置**（smoothing GridSearch
  2h22m〔68%〕+ low-rank 59m + 量化 39s；其預設 batch64 兩度 OOM，
  以數值中性 batch16 覆蓋方可在 32G 卡完成）。
- ours：cov/act 統計**零重收**（格式無關，沿用 NVFP4 產物）+
  選單 **2h10m → 11 配置**（λ7 點 + MX 域 S 增益 183/224 過門 +
  α4 點）。**格式可攜性實證**：換格式我們只付選單錢、他們全額重付。

### 選單（qdiff-128）

λ\*=0.3；**λ→∞（plain SVD 基底）在 MX 下災難崩潰**（PSNR 5.9/SSIM
0.03 vs NVFP4 端點可競爭）——格式越粗、H 基底的 outlier 吸收越
攸關。S_rms@0.25 對基底 **4:0**（+0.81dB、FID-proxy −10.9）→
定案 damp0.3+S_rms@0.25。閉式 S 在粗格式火力全開（NVFP4 邊際小）。

### 官方 MJHQ-2500（同一 MXW4A4Linear Triton kernel、同 seeds/ref/GT）

| 指標 | ours-MX | svdq-MX（其 pipeline MX 全重校準） | NVFP4 邊際對照 |
|---|---|---|---|
| PSNR ↑ | **16.539** | 15.236 | **+1.30dB**（NVFP4 +0.54） |
| LPIPS ↓ | **0.3794** | 0.4390 | **−0.060**（−0.017） |
| SSIM ↑ | **0.6283** | 0.5655 | **+0.063**（+0.013） |
| FID-ref ↓ | 28.678 | **25.342** | ✗ +3.34（NVFP4 −0.33） |
| FID-GT ↓ | 32.703 | **30.658** | ✗ +2.04（−0.18） |

**3:2**。判讀：(1) **保真邊際放大 2–4×**——「NVFP4 主表邊際小 =
基線飽和」的假說在保真面獲得確證，+1.30dB 與 DiRotQ 論文對
svdq-INT4 的宣稱同級（且我們是真 kernel + 對手全重校準的更硬設定）；
(2) **FID 兩項翻輸**：重退化域（PSNR ~15–16）中 FID 量的是分布
真實感而非保真——svdq 的逐層輸出誤差 gridsearch 保分布、我們的
配置保保真。誠實呈現為 fidelity/realism 取捨，兩法在 MX 皆遠遜
NVFP4（格式選擇 >> 校準方法）。
（附註：其 MX smoothing 力度大增〔183/224 層有 headroom〕，
其 FID 優勢與此相關；後續可選消融：ours 不同 α 的 FID-保真
Pareto 曲線。）

### 產物

`results/pixart_mx_{qdiff128,test2500}.json`、
`pixart_mx_svdq_kernel.pt`、`svdq_mx_dump/`、MXW4A4Linear（Triton，
act 量化位級=參考、記憶體 0.49 vs fp16 1.16 GiB）。
