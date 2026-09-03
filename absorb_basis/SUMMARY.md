# DiRotQ-absorb-basis 總結：方法 × 對 SVDQuant 完整戰果

版本：2026-09-03。單檔總覽（方法細節 → METHOD.md；理論證明 →
THEORY.md；逐輪記錄 → PLAN*.md）。

---

## 1. 方法一頁版

**問題**：diffusion transformer 的 W4A4（NVFP4）後訓練量化，
基線 = SVDQuant（低秩吸收 + smoothing，nunchaku kernel）。

**每個量化層的分解**（與 SVDQuant 順序相反：先分解、後選擇性平滑）：

```
y = (x·L)Dᵀ           ← rank-32 lora，永遠吃 raw x（16-bit）
  + Q₄(x/s)·Q₄(W_res·diag(s))ᵀ   ← 4-bit 主分支（NVFP4 kernel）
W_res = W − L·D，s 僅作用於殘差、逐 hook 守門（未選層 s=1）
```

| 階段 | 我們 | SVDQuant | 理論關係（THEORY.md） |
|---|---|---|---|
| 目標 | 部署誤差恆等式 ‖Δ‖²_H = ‖XΔᵀ‖²_F（T1） | ‖X‖F·‖Δ‖F 上界 | 他們的 Prop 4.1 是我們目標的 Cauchy–Schwarz 鬆弛（T4） |
| 低秩基底 | **H-SVD**：min rank-r ‖W−L‖_{H_λ}，加權 Eckart–Young 精確解（T2） | plain SVD（min ‖R‖F） | plain SVD = 我們 λ→∞ 端點（T3.1）；λ = 譜不確定球精確 min-max（T3.2） |
| 平滑 | 分解後、僅殘差、僅過門 hook；**閉式** s=(rms_x/rms_w)，α=0.5 為可分離上界精確最小元（T5） | 分解前全域 smooth（gridsearch 因子，lora 吃 smoothed x） | 先 smooth 再取基底的全家族 round-1 實測全負（已關閉） |
| 殘差量化 | **GPTQ**（OBS 逐步精確最小化同一 H 度量）on 真 NVFP4 兩級 kernel 網格、act-order | absmax 樸素捨入 | 目標對齊 vs 不對齊（T4） |
| 配置選擇 | **Algorithm 1**：λ 網格 pairwise + S×α 貪婪守門（≥3/4），全自動、只用校準資料、no-regret 回退 | 固定配方 | 家族包含 → 選擇弱優（T3） |
| 特殊層 | 全部與 SVDQuant skip 對齊；FLUX adanorm **自有 temb 校準重量化**（稽核抓到的隱藏依賴，輸出誤差比官方 AWQ 好 15×） | adanorm W4A16（AWQ 式） | — |

**零依賴**：使用者部署全程不需 SVDQuant 任何校準結果（smoothing dump、
官方 smooth、cov_actq_smooth、adanorm 位元全禁；FLUX 官方檔僅佈局容器，
稽核逐 key 驗證 2280 個校準衍生 tensors 全數替換）。

**校準協定**：qdiff-128 prompts（與 SVDQuant 相同資料、相同數量）；
選擇只看 qdiff-128 四判準 vs bf16 ref；官方測試集從不參與選擇。

---

## 2. 官方主表（零依賴配置，真 nunchaku kernel，五指標）——**29/30**

| 模型（步數/樣本） | 配置 | PSNR↑ | LPIPS↓ | SSIM↑ | FID-ref↓ | FID-GT↓ | 戰績 |
|---|---|---|---|---|---|---|---|
| **FLUX-schnell**（4/1000） | λ0.01 | **18.88** / 18.82 | **0.2299** / 0.2319 | **0.7439** / 0.7430 | **27.86** / 28.27 | **60.08** / 60.40 | **5:0** |
| **FLUX.1-dev**（50/500） | λ0.3+S_rms@0.25 | **21.46** / 21.06 | **0.1977** / 0.2111 | **0.8162** / 0.8047 | **37.83** / 39.39 | **94.35** / 94.95 | **5:0** |
| **PixArt-Σ**（20/2500） | λ0.1+S_rms@0.5 | **18.42** / 17.88 | **0.2784** / 0.2951 | **0.6852** / 0.6727 | **19.89** / 20.22 | **28.24** / 28.42 | **5:0** |
| **SANA-1.6B**（20/2500） | λ0.3+S_rms@0.25 | **19.92** / 19.76 | **0.1598** / 0.1624 | **0.7469** / 0.7426 | 10.51 / **10.41** | **27.08** / 27.22 | **4:1** |
| **SDXL-Turbo**（4/2500） | λ0.3 | **19.23** / 19.08 | **0.2140** / 0.2201 | **0.6869** / 0.6769 | **12.13** / 12.22 | **35.01** / 35.21 | **5:0** |
| **SDXL-base**（30/1000） | λ0.001 | **23.63** / 22.73 | **0.1885** / 0.2210 | **0.7959** / 0.7757 | **29.51** / 32.73 | **60.11** / 61.06 | **5:0** |

（每格「我們 / SVDQuant」；SVDQuant 側 = 官方 nunchaku 權重（FLUX 系）
或我們代跑其完整 pipeline（其餘），同 kernel 同種子同 ref/GT。）

## 3. 部署成本 parity（速度表，ms/forward，RTX 5090）

| 模型 | fp16/bf16 | ours | SVDQuant | Δ |
|---|---|---|---|---|
| FLUX-schnell（C++ 全路徑） | 455.4 | 159.0 | 159.1 | −0.1% |
| PixArt-Σ | 92.7 | 66.6 | 66.2 | +0.6% |
| SANA-1.6B | 44.7 | 36.7 | 38.0 | −3.4% |
| SDXL-Turbo | 55.4 | 90.2 | 88.2 | +2.3% |
| SDXL-base | 86.6 | 98.0 | 98.6 | −0.6% |

同 kernel、同記憶體佈局 → 部署成本相同；加速倍率取決於執行路徑
（見 PLAN.md），非本文 claim。

## 4. 校準成本

| | SVDQuant | Ours |
|---|---|---|
| 產出 | 1 配置 / 5–7h（實測） | 10–14 配置自動搜索 / 小模型 2–3.5h、dev 12B ~12h |
| 閉式 S | —（gridsearch 為其 5–7h 大宗） | **3.5–13 分/模型**（+FLUX temb 2 分） |
| 記憶體 | act 收集 O(樣本×token×d)，SDXL-base 實測 OOM | 串流 O(d²)，全程無 OOM |
| 單配置邊際 | 全 pipeline 重跑 | build 5–30 分 |

## 5. 關鍵消融（全部有檔）

1. **基底**（`svdbasis_qdiff128.json`）：plain SVD（=SVDQuant 分解）放進
   我們 pipeline，qdiff 四判準——λ\* 在 pixart +1.13dB / sana +0.35 /
   sdxl-base +0.29 / schnell +0.11 全勝；turbo/dev 偏好大 λ 端
   → λ 族 + 逐模型選擇的必要性（無固定基底全贏）。
2. **λ 網格上端擴展**（PLAN_LAMBDAEXT）：turbo/dev 校準端 λ\*→1e6，
   官方 turbo 4:1（FID-GT −0.04 翻車）、dev 5:0（FID-ref −1.45、餘毫釐）
   → λ 極端存在校準→官方泛化縫隙；**採納與否待裁決**
   （A 凍結原網格=上表 29/30 / B 採納=28/30）。
3. **S 來源**（PLAN_SELFSMOOTH）：閉式 vs svdq 借用因子——sana/pixart
   官方幾乎逐位相同；sdxl-base svdq-S 係守門偽陽性（無 S 更好）；
   schnell 借用版多 +0.21dB（唯一實質代價）；dev 閉式過門而借用版全拒。
4. **adanorm**：純權重 MSE 重量化 −0.49dB（qdiff）→ temb 加權修復至
   −0.03dB（官方分解），證明官方 adanorm 為活化感知（校準產物）。
5. **關閉路線**（負結果檔案齊全）：smooth-then-basis 全家族、
   per-channel top（3/3 負）、PCA 基底（228/228 層輸 H-SVD）、
   逐層 λ/逐層 α/lora 交替精修（headroom 分析 +0.01~0.03dB）。

## 6. 下一步（PLAN_NEXTQ，待啟動）

P1 rank waterfilling（oracle −8.3% 加權殘差能量；零校準、FLOPs 守恆）
→ P2 down-proj 閉式 S → P3 clip 重審 → P4 bias 修正 → P5 block 重建。
全部推理成本可藏 kernel。
