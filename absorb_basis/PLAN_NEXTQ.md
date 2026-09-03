# 下一波品質提升計畫（PLAN_NEXTQ）

狀態：**規劃定案，待使用者啟動**（2026-09-03）。依據：THEORY.md T1–T5、
headroom_analysis.py 離線分析（results/headroom_analysis.log）、
lambdaext 泛化縫隙觀察。

## 分析結論（headroom_analysis，FLUX-schnell）

- A 逐層 λ：活化加權下 10/12 層偏好大 λ，但端到端 schnell 是小 λ 勝
  → 逐層離線代理有系統性風險；逐層 λ 關閉，教訓寫入所有後續守門。
- B rank waterfill oracle：同預算加權殘差能量 −8.3%（rank [2,64]，
  30 層觸頂）→ 最強方向。
- C 逐層 α：+0.010dB → 關閉。
- D lora 交替精修：median +0.028dB → 關閉。
- E bias 項：輸出功率 0.2–0.6% → 近零成本可試。

## 計畫（成本標注；全部方向推理成本可藏 kernel）

| P | 方向 | 校準成本 | 推理成本 | 步驟 |
|---|---|---|---|---|
| P1 | Rank waterfilling（總 rank 守恆、cap 64） | 零（譜已算） | 零（Σr 守恆；需一次 nunchaku 任意-rank loader 驗證） | 存譜→配額器→builder --rank-map→schnell qdiff 試點→守門→六模型 |
| P2 | Down-proj 閉式 S（現 smooth=1） | +15m/模型（down act 收集） | 零（smooth 槽已存在） | 收樣→離線增益 τ 篩→選單 |
| P3 | Clip-search 重審（hdiag 加權 micro-scale 網格；舊負結論=PCA 時代，PLAN.md 已標待重審） | 零（build ×1.5） | 零 | pixart+schnell 先測→選單 |
| P4 | Bias correction 入選單（FLUX 已有旗標，移植 3 builders） | ~零 | 零（併 bias） | 移植→選單 |
| P5 | Block 重建（plan R；目標 SANA FID-ref） | +1–2h/模型 | 零 | 視 P1–P4 結果 |

守門原則：一律 Algorithm-1 端到端四判準 ≥3/4（A 教訓：不信離線代理）。
官方重跑僅配置變更模型。預估：P1 試點 ~1.5h；P1 全鋪 ~8h；
P2 ~3h；P3 ~4h；P4 ~3h。

## 執行結果（2026-09-03；P3+P2 完成，P1/P5 未動）

### 精確等價加速（先行落地，全部逐位驗證）

`--reuse-from`（增量重建：dev S 候選 build 35 分→1 分 45 秒，整檔
逐位同歷史）、`gptq_prepare`+`--factor-cache`（per-hook 置換+逆
Hessian Cholesky 快取，單元逐位同）、`--adanorm-cache`。本輪 P2 四個
候選 build 合計 <20 分（原 ~2.5h）。

### P3 clip-search 重審 — **兩模型皆「守門過、官方翻車」→ 不採納**

- pixart：qdiff 3:1 過門；官方 2500 對 SVDQuant 仍 5:0，但對現行
  配置 **1:4**（PSNR −0.09）。
- schnell：qdiff 3:1 過門；官方 1000 **對現行配置 0:5、對 SVDQuant
  1:4（PSNR 18.62 vs 18.82）**——若採納會直接輸掉記分板。
- 結論：clip 在 hsvd 配方下維持負結果（與 round-1 一致，這次含
  完整官方證據）；`{model}_p23_test*.json` 留檔。

### P2 down-proj 閉式 S — **乾淨的方法性負結果**

- 逐層增益：schnell 2/76 過 τ（median −0.19dB）、dev 2/76
  （median −0.17dB）——H-SVD lora 已吸收 GELU 輸出的 outlier 結構，
  SVDQuant 不 smooth down 層被證明合理。
- 守門：schnell α 兩點全拒；dev α=0.25 以小邊際 4:0 過門，但官方
  對現行配置 **2:3**（PSNR −0.065）→ 不採納。
- 使用者校準成本結論：down act 收集+增益 ~15 分/FLUX 模型，
  但既然負結果，正式 pipeline 不納入（零增量維持）。

### 系統性發現：qdiff-128 守門的解析度下限（四案定論）

λext-turbo（FID-proxy 反對票應驗）、λext-dev（4:0 不轉移）、
P3-clip×2（3:1 全翻車）、P2-dev（小邊際 4:0 不轉移）——
**小邊際（~±0.05dB 級）的守門通過不具官方轉移力**；歷史上轉移成功
的接受案例（S_rms於 sana/pixart、λ 選擇）皆為大邊際。論文寫法：
(1) 發表配置凍結於 selfsmooth 定版；(2) 守門政策建議升級為
「≥3/4 且邊際高於雜訊底」或 bootstrap 顯著性；(3) 本輪全部官方
數字作為守門極限的消融證據（對 SVDQuant 的 5:0/4:1 在所有變體下
從未破壞——方法魯棒性的側面證據）。

### 狀態

P3/P2 關閉（documented negatives）；P1（waterfill）與 P5（block
重建）依使用者指示待啟動；bootstrap CI 由「建議」升級為「必需」。

## 補充分析（2026-09-03 晚）：低精度分支誤差分解與置換判定

- **分解（lowbranch_headroom.log，schnell 24 層）**：4-bit 分支誤差
  中位 **75% 來自活化量化、25% 來自權重**；GPTQ 已對權重項貢獻
  +4.57dB（vs RTN），oracle-H 再擠僅 +2.12dB（權重項）；
  **權重誤差歸零的層 QSNR 天花板 = +1.27dB**。所有權重側方法
  （P1/P5/clip/bias）合計不可能穿過此 floor——P2/P3/λext 不轉移
  的機制級解釋。
- **置換（rotation 家族唯一 kernel-free 成員）判定
  （perm_headroom.log）**：量級排序分組 main 層噪聲級（median
  +0.04dB）、down 層**有害**（act 項 median −1.9dB、個別 −8dB）——
  動態逐 token act 量化下「同量級分組」直覺失效（大通道共享粗
  scale 互傷）。方向關閉。
- **蓋棺**：剩餘誤差鎖在部署格式的活化量化器內稟誤差；對角縮放與
  重分組皆不可及。論文結語素材：「pipeline 運行於 NVFP4 格式的
  有效天花板」。P1/P5 的期望值因 +1.27dB floor 大幅下修，建議
  降級或僅作消融行。

## 補充分析 2（2026-09-03 深夜）：rotation 全家族判定——關閉且有害

block-16 {Hadamard, RHT, Haar}、DuQuant 式 zigzag+RHT、以及**全維
Haar oracle（不可部署上界）**在 act 樣本上全數為負
（`rot16_headroom.log`）：main median −0.18~−0.29dB、down
−1.5~−2.6dB（act 項；權重 RTN 項亦全負）。機制：incoherence 收益
屬於粗 scale 格式（per-channel INT4 / MXFP4 group-32 二冪）；
NVFP4 動態 per-group-16 e4m3 已領走稀疏紅利，旋轉抹平尖峰反而
拉高總誤差。定位句：「rotation 修補粗格式；細格式 + outlier 吸收
分解使其多餘且有害」——同時解釋 DiRotQ 論文（INT4）與 KroQuant
（MXFP4）在其格式上有收益的原因。act 項 75% 的不可約性證據鏈
完整：S → 置換 → block 旋轉 → 全維 oracle 全數排除。

## 補充分析 3（2026-09-04 凌晨）：噪聲感知基底——完整弧線後關閉

思路：殘差分支在權重精確時的誤差恆等於 tr(W_res·G_E·W_resᵀ)
（G_E=act 噪聲 Gram）→ 基底度量應為噪聲而非訊號（H 只是其代理，
λ 阻尼是粗修正——順帶解釋 turbo/dev 的 λ→∞ 傾向）。驗證弧線：
(1) 同樣本 oracle：main +0.19 / down +0.63dB（`noiseaware_oracle.log`）；
(2) split-half：全 G_E 塌到 +0.06（過擬合），**diag(G_E)/收縮版存活
+0.14~0.22dB、18 層全正**（`ge_splithalf.log`/`ge_shrink.log`）——
且與 Prop 4′ 對照鮮明：diag(H) 有害、diag(G_E) 全正，關鍵是度量
「量什麼」而非對角與否；(3) 端到端（`--basis-metric noise-diag` 已
實作入 builder）：schnell qdiff 候選對基底 **1:3 落敗**。結論：
逐層 ~0.15dB 級增益不轉移（與本週全部案例一致），方向關閉；
理論觀點（lora 遮噪聲/GPTQ 管訊號的解耦、H 作為 G_E 代理）保留為
論文討論素材。**品質優化戰役至此收官：方法在 NVFP4 格式有效天花板
確立，剩餘工作全屬論文支撐類。**
