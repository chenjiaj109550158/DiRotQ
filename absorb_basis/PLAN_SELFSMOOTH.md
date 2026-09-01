# 去除 SVDQuant checkpoint 依賴：自足式 smoothing / 品質提升方案

狀態：**規劃中，未開始實作**（2026-09-01 撰寫；使用者指示先存檔，
之後再啟動實作測試）。

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
