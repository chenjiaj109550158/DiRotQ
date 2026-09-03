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
