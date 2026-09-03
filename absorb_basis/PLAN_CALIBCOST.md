# 校準成本正式量測計畫（PLAN_CALIBCOST）

狀態：**記錄待執行**（2026-09-03 撰寫；使用者指示：最終方法定案後
才執行——因加速機制與最終配置會改變 ours 側數字，先測會白測）。

## 目標與定位

產出論文級的 calibration 成本對比表。**定位為 secondary
contribution，主張「可配置性與邊際成本」而非「總時更快」**（誠實
前提：dev 12B 全程 ~12h，不小於他們小模型的 5–7h）。四個可辯護
claim（皆有/將有實測）：

1. 同預算產出量級差：SVDQuant 5–7h 產 1 配置 vs ours 同級時間產
   10–14 配置全自動搜索；單配置邊際成本 2–6 分鐘（--reuse-from
   增量重建，逐位驗證）。
2. 記憶體可行性：串流 O(d²) vs 他們 O(樣本×token×d) 無 offload
   （SDXL-base 128 樣本 OOM 已記錄；FLUX 探針補強）。
3. 閉式 S 分鐘級 vs gridsearch smoothing 小時級（品質 29/30 保持）。
4. 零依賴：無需任何 SVDQuant 校準產物。

## 執行步驟（估：~半天工作 + ~5h GPU）

| # | 步驟 | 成本 | 說明 |
|---|---|---|---|
| A | 日誌考古 | 半天、零 GPU | 六模型 ours 全程分段時間（selfsmooth/fluxdev/sdxl30 等 chain log 時間戳）+ 4 模型 svdq 側實測（pixart/sana/turbo/sdxl-base，deepcompressor log 含 smoothing/SVD/GPTQ 分段；sdxl-base 7h10m 已精確記錄）。同機 RTX 5090、同 caches |
| B | FLUX svdq 可行性探針 | ~1h GPU | 以 deepcompressor 對 schnell 起跑其校準，記錄 act 收集階段 RAM 行為至 OOM/完成擇一；不求跑完，求可行性數據點 |
| C | 加速版從零重計時 | ~4h GPU | pixart 全 pipeline 乾淨重跑一次（caches→cov→向量→加速選單→build），作為「從零部署使用者」的端到端錨點數字 |

## 前置條件（為何等最終方法）

- 最終配置若再變（P1 waterfill / λext 採納決策 / MXFP4 輪），選單
  內容與 build 次數隨之變 → C 的數字會過期。
- 加速（reuse-from / factor-cache / adanorm-cache）已落地且逐位驗
  證，C 應以加速後 pipeline 計時。
- bootstrap CI 若改變守門政策（邊際門檻），選單成本亦受影響。

## 產出

`results/calibcost_table.json` + SUMMARY.md 成本節改版 +（論文）
分段成本堆疊圖：caches（共同前置）/ cov / 閉式 S / 選單 / 官方，
ours vs svdq 並排，附記憶體峰值欄。

## 相關既有記錄

PLAN_SDXL30.md（svdq 7h10m、OOM 事件）、PLAN_FLUXDEV.md（ours dev
全程 22.5h 含官方）、PLAN_SELFSMOOTH.md 成本節、PLAN_NEXTQ.md
加速實測（35 分→1 分 45 秒）。
