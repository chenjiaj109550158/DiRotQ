# PLAN_PATHA_ALL：六模型全量 Path A 防混用驗證

狀態：執行中（2026-09-04）。動機（使用者）：歷輪實作方法眾多、早期
甚至依賴 SVDQuant checkpoint，須排除「方法混用」導致 release 權重
（Path B）≠ 從零校準（Path A）。

## 兩層驗證的覆蓋範圍

| 檢測 | 蓋到 | 蓋不到 |
|---|---|---|
| quickcheck（原統計→新 repo build→位級比 release） | 程式碼/flags 漂移、build 混用 | **輸入統計本身的污染** |
| **全量 Path A**（本計畫） | 一切：caches→統計→向量→build 全鏈重生，逐級位級比對 | —（金標準） |

pixart 已過金標準（cov 196 / vectors 168 / ckpt 1568 條全位級）。

## 每模型流程（gate 逐級，任何一級不合立即停在該級=污染定位）

1. **collector 忠實性**（單 prompt 位級 vs vault/DC 協定 caches）——
   已證：pixart 40/40、sdxl 4/4；本輪補 sana、flux-schnell、flux-dev。
2. caches 全量重生（workdir）
3. 統計重收（cov / cov_conv / flux: cov+cov_down+act_samples+act_amax+temb）
   → **位級比對原統計檔**（資訊級，定位污染源）
4. 向量重推（sana；dev）→ 位級比對
5. build（凍結 config）→ **位級比對 hf_staging release 檔（gate）**
6. flux 另加：容器稽核（所有校準衍生 tensor ≠ svdq 官方容器）
7. 清理該模型 workdir caches（磁碟輪轉）

## 估時（GPU 串行，接在 quickcheck 鏈後）

| 模型 | caches | 統計 | build | 小計 |
|---|---|---|---|---|
| sana | ~1h | ~1h | 15m | ~2.5h |
| sdxl-turbo | ~10m | ~20m | 30m | ~1h |
| sdxl-base | ~2h | ~2.5h | 30m | ~5h |
| flux-schnell | ~2h | ~5h（cov_down 大宗） | 1.5h | ~8.5h |
| flux-dev | ~5h | ~6h（caches_sub 協定） | 2h | ~13h |

合計 ~30h。磁碟：逐模型清 caches；flux cov_down 產物直接在 workdir
比對後刪除（vault 原檔為準）。

## 失敗處置（預先聲明）

任何一級位級不合 = 找到混用點：回報差異層級與內容，**以乾淨 Path A
產物重建該模型 release + 重跑其官方基準**（數字以乾淨版為準），
不掩蓋。
