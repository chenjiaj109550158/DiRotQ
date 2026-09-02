# 去除 SVDQuant checkpoint 依賴：自足式 smoothing / 品質提升方案

狀態：**完成**（2026-09-02 全鏈執行完畢；六模型零依賴版 29/30 保持，
校準成本與消融見文末「執行結果」）。

## 背景與動機

現行五模型最終配置中 4 個含 S（選擇性 smoothing），其 s 因子來自
SVDQuant 的校準產物（FLUX 系解包官方 checkpoint 的 packed smooth；
PixArt/SANA/SDXL 取自我們代跑的 deepcompressor smoothing dump，
每模型 5–7h）。核心方法（H-SVD basis + GPTQ + λ）零依賴，但含 S 的
配置使「校準更便宜」的主張不乾淨，reviewer 可追問
「離開 SVDQuant 產物還剩多少」。

**現有結果的關鍵提示**：α<1 系統性勝出（PixArt 0.5、SANA 0.25、
SDXL-base 0.5）→ gridsearch 精調因子對我們的「殘差量化」設定過強，
真正需要的是「方向正確的每通道尺度 + α 強度旋鈕 + 守門」。
精調不必要 → 閉式因子大概率足夠，甚至更匹配。

## 方案 1（首選）：H 對角線 RMS smoothing——零額外收集

- **關鍵觀察**：cov 對角線 `diag(H) = Σxᵢ²/n` 就是每通道活化均方值，
  活化側統計已經在手，零額外收集。
- **因子**：`s_k = (rms_{x,k})^α / (rms_{w,k})^{1−α}`，其中
  `rms_{x,k} = sqrt(diag(H)_k)`、`rms_{w,k}` 取 **W_res(λ*) 的每輸入
  通道 RMS**（殘差感知：平衡對象是實際被量化的殘差，非原權重 W）。
- α 沿用現有網格 {0.25, 0.5, 0.75, 1.0} + 四判準守門（≥3/4），
  機制零改動，只換 s 來源。
- 層選擇沿用現有離線增益量測（`measure_smooth_gain_flux.py` /
  `measure_smooth_gain_sdxl.py` 均接受任意 s，act samples + cov 皆已有）。

**理論錨點（依賴版所沒有的）**：對輸出誤差的可分離上界
`‖ΔY‖² ≤ (Σₖ rms²_{x,k}/s²ₖ)·(Σₖ rms²_{w,k}s²ₖ)`，
逐通道駐點解為 `s_k = (rms_{x,k}/rms_{w,k})^{1/2}` —— 即 α=0.5 閉式解
是該上界的**精確全域最小元**（AM-GM），且界對的正是 W_res。
SVDQuant 的 gridsearch 因子只在「他們的目標、無殘差分解的設定」下
逐層經驗最優，對我們的設定無最優性——這解釋了 α<1 勝出的現象。

## 方案 2：殘差感知 SmoothQuant（amax 版）

- 經典式 `s_k = amax_{x,k}^α / amax_{w,k}^{1−α}`；amax_x 用
  `collect_act_amax.py`（已存在，qdiff-128、幾分鐘），amax_w 取 W_res。
- 與方案 1 差異：absmax 對單 token outlier 敏感；可用已收 act samples
  的 99.9 percentile 折衷（穩健版）。
- 可與方案 1 同場守門排名（同一 menu 放兩族候選），或先各測 α=0.5
  單點篩方向再展開。

## 方案 3：校準均值 bias correction（免 smoothing 的正交槓桿）

- `build_checkpoint --bias-correct` 已實作（act samples 均值修正
  量化偏差），只用我們的資料、成本近零，從未進過正式選單。
- 掛進 Algorithm-1 當一個新守門項（每模型一個 build + qdiff-128 排名）。
- 與 S 正交、可疊加；PixArt/SANA/SDXL builder 尚無此旗標，如要
  全模型套用需小幅移植（FLUX 已有）。

## 依賴稽核（三方案共通）

校準產物零依賴：s/守門/排名全部只用我們的 cov、act_amax、act samples
與 qdiff-128 prompts（出處 Q-Diffusion）。須在論文披露的三個
「非校準產物」接觸點：(1) nunchaku kernel 為 parity 主張刻意共用；
(2) FLUX build 以官方檔為佈局容器，但所有量化層 tensors 全數替換，
剩餘為 BFL 原始 bf16 權重；(3) deepcompressor 開源框架僅作實驗
基礎設施（等同用 PyTorch）。

## 理論保證的邊界（論文措辭依據）

1. 對「無 S 基底」：守門機制自帶校準集 no-regret（不過門即回退）。
2. 方案 1 α=0.5：上界精確最小元 + 殘差感知（見上）。
3. 對「借用因子版 S」：逐點保證不存在；超集論證
   （menu ⊇ {無S, 閉式, svdq}）只用於**消融表**，正式 pipeline 不放
   svdq 候選以維持零依賴。測試集優劣以消融經驗證據呈現，
   現有 α<1 證據對閉式版有利。

## 驗證設計與成本

三方比較（qdiff-128 四判準）：S_closed(α*) vs 現行 S_svdq(α*) vs 無 S。

| 步驟 | 成本/模型 |
|---|---|
| 因子計算 | ~0（方案 1）/ 幾分鐘（方案 2） |
| 增益量測 + 守門層選擇（離線） | ~10 min |
| α 網格 build+gen+rank（沿用選單機制） | PixArt/SANA/Turbo ~1–1.5h；SDXL-base ~1.5h；FLUX 系各 ~3.5h |
| 官方重跑 | 僅配置改變時 |

**試點**：SANA（最快、α=0.25 最敏感、唯一 4:1 模型）先跑方案 1+2 的
α=0.5 單點篩選 → 決定全面鋪開順序。

**判讀**：閉式持平或勝 → 五模型 pipeline 對 SVDQuant 零依賴，校準成本
= cov + 選單（每模型 1.5–3.5h），主張完全乾淨；閉式略輸 → 論文給
self-contained / borrowed-factors 兩版數字，依賴降級為可選增強。

## 實作注意（動工時）

- PixArt/SANA/SDXL builder 需加「s 來源＝外部向量檔」的通用入口
  （現在只吃 SVDQuant smooth.pt 格式；建議統一成
  `--smooth-vectors <pt: {hook_key: s}>`，svdq/closed 兩種來源都轉成
  此格式，builder 邏輯不分支）。
- 方案 1/2 的 s 計算腳本要「先解 λ* 的 W_res 再取 rms/amax」，與
  增益量測共用 H-SVD 呼叫（一次算完存檔，避免重複分解）。
- FLUX 側 build_checkpoint 的 `--select-smooth-gains` 路徑目前從
  `--official` 解包 s；需加 `--select-smooth-vectors` 讓 s 改讀外部檔。
- 守門與 α 網格完全沿用 round-3 定版流程，結果檔命名
  `{model}_selfsmooth_qdiff128.json`。

## 執行計畫定版（2026-09-02，動工）

**範圍**：六模型（flux-schnell λ0.01、flux-dev λ0.3、pixart λ0.1、
sana λ0.3、sdxl-turbo λ0.3、sdxl-base λ0.001）。λ\* 固定沿用
（λ 選擇本就零依賴）；本輪只重做 S 階段，menu = {無S 基底,
S_closed(rms)@α, S_closed(amax)@α}，**不含任何 SVDQuant 產物**。

**強化原則（2026-09-02 使用者裁定）**：不只官方 checkpoint 的 smooth
因子，**連我們自己代跑 deepcompressor smoothing 產生的
`svdq_model_dump/`（smooth.pt/scale.pt/model.pt）與
`cov_actq_smooth.pt`（在 svdq-s 域收的 cov）也全面禁用**。判準：
使用者要把 ours 部署到一個新模型時，全程不需要執行或讀取 SVDQuant
的任何 calibration 結果。builder 對 smoothed 層一律走解析
`H/(s⊗s)`，不傳 `--gptq-cov`。消融表中的 S_svdq 欄位僅沿用既有
round3 JSON 作對照呈現，不作任何新 pipeline 輸入。FLUX pilot build
後加稽核步：逐 key 驗證輸出檔中所有校準衍生 tensors
（qweight/wscales/smooth/smooth_orig/lora_down/lora_up/wtscale）
與官方容器不同（全數被我們覆寫）。

**基礎設施盤點**（2026-09-02 實測）：
- 三個 kernel builder（pixart/sana/sdxl）已吃通用 `--smooth-pt
  {skey: s}` + `--gains` json → 零改動，只換輸入檔。
- FLUX `build_checkpoint.py` 需加 `--select-smooth-vectors <pt>`
  （s 改讀外部檔，取代官方 checkpoint 解包；其餘 gate/α 機制不動）。
- qdiff-128 基底圖與 ref 圖六模型皆在 runs/（λ\* 基底免重生）。
- caches 現況：sdxl-turbo/sdxl-base/flux-dev 在；**pixart/sana 已因
  磁碟清理刪除 → 用各自 collect_calibration_dataset.py 重生**
  （defaults = 20 步 g4.5 1024px qdiff-128，與原始 cov 收集同協定）；
  flux-schnell caches 亦刪但離線量測只需 act samples/amax（皆在）。

**s 定義（存全強度 s，builder 套 s^α）**：
- rms 族：`s_k = rms_x,k / rms_w,k`；rms_x=sqrt(diag H)（cov 免費），
  rms_w = W_res(λ*) hook-併排的每輸入通道 RMS。α=0.5 即理論閉式解。
- amax 族：`s_k = amax_x,k / amax_w,k`；amax_x 全流收集
  （schnell 用既有 absorb_act_amax.pt；dev 對 caches 跑 collect_act_amax；
  pixart/sana/sdxl 於 hook pass 全 caches 累積），amax_w=W_res 每通道絕對最大。
- 兩族皆做幾何平均=1 正規化（全域尺度對 NVFP4 兩級 scale 近乎不變，
  正規化讓 α 旋鈕語意乾淨）；smoothing 範圍與現行 S 完全相同
  （flux down-proj、sdxl conv、cross-KV 不 smooth）。

**增益量測**（gate 層選擇，τ=+0.3dB 不變，s 取 α=1）：
- flux 系：離線（act samples + cov），同 measure_smooth_gain_flux 法。
- pixart/sana/sdxl 系：hook 式 16 檔 strided forward（sdxl 先例），
  兩族 s 同 pass 各算 e1。

**新增/修改檔案**：`selfsmooth_vectors_flux.py`（s+增益，schnell/dev）、
`selfsmooth_vectors_hook.py`（pixart/sana/sdxl 系：pass1 全 caches amax
→ s 兩族 → pass2 增益）、`build_checkpoint.py`（--select-smooth-vectors）、
`selfsmooth_driver.py`（round3_driver 同款 4 判準 wins/stats + 貪婪守門）、
`run_selfsmooth_all.sh`（setsid 鏈）。

**流程**：(0) pixart/sana caches 重生 + dev act_amax →
(1) 六模型 s 向量 + 增益 → (2) SANA 試點：rms@0.5 vs amax@0.5 vs
基底（qdiff-128 四判準）→ 決定族別鋪開 → (3) 六模型勝族 α 網格
{0.25,0.5,0.75,1.0} 貪婪守門（flux 落選 safetensors 即刪控磁碟）→
(4) 消融三方表（closed vs svdq-S vs 無S；svdq 側全部沿用既有 round3
JSON，不重跑）→ (5) 最終配置 ≠ 現行官方配置的模型重跑官方 benchmark
（僅 ours 側生成；ref/svdq/GT 圖皆在）→ (6) 校準成本回報。

**預估**：caches 重生 ~1.5–2h；向量+增益 ~2h；試點 ~40m；α 網格
六模型 ~12h；官方重跑（預計 4 模型：schnell/pixart/sana/sdxl-base）
~6h。合計 ~22–26h GPU。磁碟峰值 +~30G（guard 45G）。

## 稽核發現：adanorm 隱藏依賴（2026-09-02 執行中）

容器稽核在 flux 首建即中止：**76 層 adaLN modulation linears
（double norm1/norm1_context、single norm.linear）的
qweight/wscales/wzeros 一直沿用官方容器值**——SVDQuant 的 W4A16
（對稱 int4、zp=7、g64；wzeros=−7·wscales 逐位驗證）。其 scale
無法以純 min/max 或 absmax RTN 逐位重現（可能含 scale search），
無法證明 data-free → 依嚴格原則必須替換。另一類標記
（down-proj smooth 全 1）為假陽性：官方不 smooth 12288 維層、
我們亦寫 1，單位元素不含校準資訊，稽核放行。

**修復**（已實作）：
- `build_checkpoint.py` 新增 `requant_adanorm`（預設啟用，
  `--keep-official-adanorm` 保留舊行為）：對稱 int4 zp=7、每組 64
  的 MSE grid scale search（只用 bf16 權重，決定性 data-free），
  以 deepcompressor 官方打包函數輸出。單元驗證：形狀/dtype 全合、
  **重建 MSE 低於官方**（1.01e-5 vs 1.25e-5 / 6.7e-6 vs 8.4e-6）。
- flux/fluxdev 選單改以自足基底（重建無S base + 重生基底 qdiff 圖），
  且無論守門結果**一律重跑官方**（adanorm 位元已變，發表數字不可沿用）。
- 稽核規則：smooth/smooth_orig 相等且全 1 → 放行；其餘任何
  校準後綴 tensor 與容器相等 → 中止。
- pixart/sana/sdxl 部署不經 nunchaku 容器（自建 kernel .pt +
  原始 fp16 模型，adaLN 保持 fp16 原權重），無此問題。

**temb 感知修復（同日）**：純權重-MSE 重量化讓 flux 基底 qdiff 掉
−0.49dB（權重誤差更低、端到端更差 → 官方 scale 為活化感知）。改為
`collect_temb_flux.py`（qdiff-128 prompts 的 CLIP-L pooled × 32 點
timestep 網格 → silu(temb) 4096 樣本，~2 分鐘/模型，零 SVDQuant）+
活化能量對角加權的非對稱 (min,max) 收縮網格搜尋，打包與
deepcompressor 轉換器 oracle 逐位對拍。實測：temb 樣本上輸出誤差比
官方好 15×；基底 qdiff 收復至 −0.12dB（官方測試集上 adanorm 差異
僅 −0.029dB，見 schnell 分解）。另兩個稽核規則修正：全 1 smooth
（單位元素）與單元素純量巧合豁免。

## 執行結果（2026-09-02，全鏈 05:31–18:47，~13.3h GPU）

### 六模型零依賴定案與官方戰績——**29/30 保持**

| 模型 | 定案（零依賴） | 發表配置 | 官方（vs SVDQuant） |
|---|---|---|---|
| SANA | damp0.3+**S_rms@0.25** | +S_svdq@0.25 | **4:1** 保持；與借用版差小數第三位（`sana_selfsmooth_test2500.json`） |
| PixArt | damp0.1+**S_rms@0.5** | +S_svdq@0.5 | **5:0** 保持（`pixart_selfsmooth_test2500.json`） |
| SDXL-Turbo | damp0.3（S 全拒） | damp0.3 | 配置未變，免重跑 |
| SDXL-base | damp0.001（S 全拒） | +S_svdq@0.5 | **5:0** 保持且**五項全優於發表版**——svdq-S 在官方集根本沒賺（`sdxlb_selfsmooth_test1000.json`） |
| FLUX-schnell | damp0.01（S 全拒） | +S_svdq@1.0 | **5:0** 保持（PSNR 邊際 +0.06dB；分解：少 S_svdq −0.184dB、adanorm 差異 −0.029dB）（`flux_selfsmooth_test1000.json`） |
| FLUX.1-dev | damp0.3+**S_rms@0.25**（守門 4:0） | damp0.3（無 S） | **5:0** 保持（PSNR +0.40、FID-ref −1.56）（`fluxdev_selfsmooth_test500.json`） |

### 科學觀察

1. **閉式 S 與 svdq-S 的互補翻轉**：dev 上 svdq-S 四點全拒、
   S_rms@0.25 卻 4:0 過門；schnell 相反（svdq-S@1.0 有效、閉式全拒）。
   sana/pixart 閉式在與 svdq 版相同的 α 過門且官方幾乎同分。
   α=0.25 甜蜜點（sana/dev）與 0.5（pixart）再現「軟化強度 + 守門」
   的核心機制價值。
2. **sdxl-base 的 svdq-S 是守門的偽陽性**：qdiff-128 過門但官方
   1000 張五項全輸無 S 版——零依賴版反而修正了這點。
3. **schnell 是唯一有實質代價的模型**（−0.21dB，86% 來自 S）；
   borrowed-S 以「可選增強」呈現。

### 校準成本（實測；SVDQuant 對照為我們代跑 deepcompressor 的實測）

| 成分 | 成本 |
|---|---|
| caches（兩法共同前置） | pixart 5m40s / sana 20m / sdxl 系 ~0.5–1h / schnell ~40m / dev 2h04m |
| cov 串流收集（ours） | 小模型 ~1–2h；dev 4h59m；記憶體 O(d²) 無 OOM（svdq 側 O(樣本×token×d)，sdxl-base 實測 OOM） |
| **閉式 s + 增益量測（本輪新增）** | **3.5–13 分鐘/模型**；flux temb 收集 +2 分鐘 |
| dev act-amax（caches_sub） | 11m |
| S×α 選單（4 點守門） | sana 17m / pixart 25m / turbo 28m / sdxlb 56m / schnell ~1h10m / dev 2h22m |
| 官方重跑（配置變更 5 模型） | sana 58m / pixart 1h28m / sdxlb 1h04m / schnell 24m（基圖快取）/ dev 1h16m |
| SVDQuant 對照 | **5–7h/模型產 1 配置**（smoothing gridsearch 為大宗）；借因子路線需先付這筆 |

結論：去依賴把「借用 SVDQuant 5–7h smoothing 產物」換成「自己
~10 分鐘閉式計算」，品質 4 模型持平、1 模型（sdxl-base）更好、
1 模型（schnell）−0.21dB。使用者部署 ours 全程接觸的校準產物 =
qdiff-128 caches + 我們的 cov/amax/temb/s，零 SVDQuant。

### 事件記錄

執行中兩次守門級中止皆為容器稽核正確觸發：(1) adanorm 隱藏依賴
（見上節，修復後品質更好）；(2) fluxdev 單一 wtscale 純量巧合
（豁免規則）。另修 driver 重放時勿重建已刪落選檔（圖在即排名）。
磁碟壓力時釋放 pixart/sana caches 27G（vault 5120/5120 驗證後）。
