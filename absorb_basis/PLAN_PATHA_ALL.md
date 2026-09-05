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

## 事件記錄：flux-dev quickcheck 揪出 λext 殘留覆寫（2026-09-04）

quickcheck 六座 5 過 1 不合：flux-dev 2908 條中 211 條不合，**全部**
集中在 S 守門 hook 的 smooth/smooth_orig 及其下游（qweight/wscales/
wtscale）；lora/基底 2697 條全數一致 → 污染範圍精確限定在 selfsmooth
向量。時間線：release 建於 09-02 15:29；λext 06:11 先備份原始向量為
`_lam0.3` 後綴，23:20 以 λ=1e6 實驗向量覆寫無後綴檔。以 `_lam0.3`
備份重建 → **VERIFY_OK 2908 條、digest == release**。結論：release
本身乾淨（cov + lam0.3 向量 + 凍結 flags 的決定性產物），不合源自
研究磁碟的實驗殘留佔用正名。處置：flux-dev 與 sdxl-turbo（同輪同害）
的無後綴向量檔已復原為 lam0.3 原件、λext 殘留改名 `_lam1e6`；
AbsorbQuant Path A 不受影響（configs 向量步驟寫死 --lam 0.3 重算）。
quickcheck 至此 **6/6**；佇列（指標噪聲 + 五模型全量 Path A）已解鎖。

## Path A 執行結果與第二事件：機損前產物（2026-09-04 晚）

Path A 鏈實跑結果：**sdxl-turbo、sdxl-base 全鏈金標準通過**（含 cov
位級全同、GATE_OK）；sana GATE_FAIL、flux-schnell collector 對拍不合。
診斷定案（根因=08-30 機器損失事件）：
- schnell：vault caches（08-27）為**前一台機器**產物——同種子的 CUDA
  RNG 在不同 GPU/env 給不同初始 latent（step0 latent maxabs 6.5）、
  bf16 T5 嵌入整體不同（2.66）。今日 collector 兩次獨立跑位級全同
  （自身決定性無誤），但**任何程式碼都無法位級重現前機產物**。
- sana：caches 是機損後重生的（collector 對拍 vs vault 全同 ✓），但
  **stored cov 來自機損前的 DC dataset**（已刪）→ cov/向量微偏
  （多數 key maxabs ≤1e-2）→ ckpt 位級不合。
- 對照組：pixart/sdxl 的 DC dataset 與 dev 的 caches 均為機損後本機
  產物 → 全數位級通過，分界完全吻合。
處置（政策執行中，rebuild_chain）：sana 乾淨重建檔已建
（workdir patha.pt）→ 官方 2500 重跑；schnell 全 Path A 重建（今日
乾淨統計）→ 官方 1000 重跑；dev 全 Path A + release gate（統計為
機損後產物，預期位級通過）。舊 release 檔保留至數字確認後由使用者
裁定替換。

## flux-dev 金標準收官 + vectors config bug（2026-09-05）

遠端機器交叉驗證揪出第二個同類 config bug：`configs/flux-dev.yaml` 的
vectors 區塊漏傳 --model-id → 預設 schnell → **混模向量**（dev 統計 ÷
schnell 權重；架構同形不報錯、輸出貌似合理——log 掃錯誤訊息抓不到，
唯位級比對可辨）。連鎖澄清：本機稍早的「向量非決定性」判讀**撤回**
——當時比的是混模 fresh#1 vs 正版 fresh#2。仲裁：dev 權重從零向量
== 存檔 release 向量 **228 條位級全同**（向量計算實為決定性；原始
run_selfsmooth_all 有傳 --model-id）。修復 config 後重 build →
**gate 位級通過：2908 條全同、digest == release（657643a8…）、
容器稽核 2280 條全替換**。dev 全 Path A（caches→cov/cov_down/act/
temb→向量→build）金標準成立，無需重 evaluation。

**Path A 記分板：pixart ✓ / sdxl-turbo ✓ / sdxl-base ✓ / flux-dev ✓
（4/6 位級金標準）；sana、flux-schnell = 機損前統計，乾淨重建+官方
數字已備，等 A/B 裁定。**

## 裁定（2026-09-05，使用者）：B——保留原 release

sana/flux-schnell 維持原權重與主表數字（29/30）；乾淨重建版
（sana_clean 4:1 / fluxs_clean 4:1）作為魯棒性佐證存檔
（results/{sana_clean_test2500,fluxs_clean_test1000}.json + vault 權重）。
公開文件（README + REPRODUCIBILITY）如實註記兩模型統計出生於機損前
機器（vault 存檔）、本機從零為數值等價、其餘四模型全鏈位級。
HF 上傳沿用現 staging（即原 release），無需重傳。

## 更正（2026-09-05，使用者）：裁定改為 A——採用新權重

上一節 B 為口誤，最終裁定 **A**：sana/flux-schnell release 換用本機
乾淨重建版（sana patha.pt digest f6adf74c…；schnell_clean digest
fe811bd1…），主表採其官方數字（兩座各 4:1，大表 28/30）。至此六
release 全數為本機從零產物、Path A 位級可重現（兩座之收集/統計/build
決定性皆已實證）。configs digests、README/REPRODUCIBILITY、hf_staging
已同步；HF 上傳以新檔重啟。

## 跨機交叉印證（2026-09-05）

第二台獨立 RTX 5090（driver 595.71.05 ≠ 本機 595.84）以公開 repo 全
Path A 從零（自收 caches/cov/cov_down/act/向量/build）→ flux-dev
digest **657643a8… 與 release byte 全等**。跨機位級重現成立；
外殼實證可容忍小版本驅動差（釘 torch + 同 GPU 世代為要件）。

## 終驗收官（2026-09-05）：金標準 6/6 獨立實測

sana、flux-schnell 各做完全獨立 Path A 重跑（workdir 清空、caches 從
128 prompts 重收、統計/向量重算、重 build）→ digest 皆與 release 全等
（sana f6adf74c… / schnell fe811bd1…，schnell 容器稽核通過）。合併
先前四座與 flux-dev 跨機印證：**六 release 全數「從零→位級」獨立實測
成立**。驗證戰役就此收官。
