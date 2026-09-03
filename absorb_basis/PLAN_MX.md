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
