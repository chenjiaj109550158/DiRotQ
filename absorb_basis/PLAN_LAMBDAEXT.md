# λ 網格上端擴展：turbo / dev 重跑（Algorithm-1 grid top-end）

狀態：**執行中**（2026-09-02 使用者裁定）。動機：svd-basis 消融
（THEORY.md 實驗閉環）顯示兩個 λ\* 頂到網格上緣（0.3）的模型，最優
點在網格之外——turbo 1:3、dev 0:4 輸給 λ→∞ 端點。其餘四模型
λ\* 對 ∞ 全勝（3×4:0、1×3:1），不在本輪範圍。

## 設計

- **新網格點**：λ ∈ {1, 10, 1e6}（1e6 ≈ ∞ 端點；其 qdiff 圖已在
  svd-basis 消融生成，直接沿用）。
- **Stage A′ 排名**：候選 = {現任 λ\* 基底} ∪ {1, 10, 1e6}，
  qdiff-128 四判準 pairwise 積分。原網格下緣五點在 round3 已於同協定
  輸給 0.3，不重排（引用）。dev 側全部候選皆為 temb-adanorm build
  （零依賴版一致性；svdinf 圖即是）。
- **Stage C**：若 λ\* 更新 → 以新 λ\* 重算閉式向量+增益
  （s 依賴 W_res(λ\*)），S_rms × {0.25,0.5,0.75,1.0} 貪婪守門。
- **官方**：最終配置 ≠ 現行零依賴配置才重跑（turbo MJHQ-2500、
  dev MJHQ-500），vs SVDQuant + 併記前任零依賴配置作三方對照。
- 全程零 SVDQuant 校準產物；dev build 稽核照常。

## 預估

| 步驟 | turbo | dev |
|---|---|---|
| λ1、λ10 build+gen（1e6 圖快取） | ~35m | ~1h50m |
| 排名（含快取圖統計） | ~15m | ~20m |
| 新 λ\* 向量+增益 | ~12m | ~4m |
| S 選單 4α | ~1h10m | ~3h40m |
| 官方（配置變更時） | ~1h15m | ~1h35m |
| 合計 | ~3.5h | ~7.5h |

產物：`results/{sdxl,fluxdev}_lambdaext_qdiff128.json`、
`{...}_lambdaext_test{2500,500}.json`；chain
`run_lambdaext.sh`，log `results/lambdaext_chain.log`。
磁碟 237G 起跑（dev cov_down 43G 已預留在本地）。
