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
