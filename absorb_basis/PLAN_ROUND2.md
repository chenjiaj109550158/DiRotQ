# 待辦計畫（第二輪品質優化）：PSNR 與 FID 提升

狀態：**已規劃、未實作**（2026-08-29 存檔）。
基準：嚴格協定最終配置（見 PLAN.md）——FLUX λ=0.01 五項全勝 SVDQuant；
PixArt λ=0.1 近打平（PSNR/SSIM 贏、LPIPS/FID 小輸），PSNR 領先幅度小
（PixArt +0.11、FLUX +0.09）。

所有計畫共同約束：
- 校準資料 = SVDQuant 同款 qdiff 128 prompts（同資料、同量），選擇一律
  qdiff-128 端到端排名（FID proxy 納入判準）→ 冻結 → 測試集只跑一次。
- 部署不變：nunchaku SVDQW4A4Linear kernel、rank 32、速度/記憶體與
  SVDQuant 逐 bit 相同。產物只能是「同格點上的不同 codes / 不同 fp16
  側張量」。

---

## 支持數據（小型實驗，2026-08-29）

1. **誤差預算**（PixArt qdiff-32 端到端，damp0.1）：
   - weight-only 量化：19.14 dB PSNR → weight 側佔總 MSE ~68%
   - full：17.45 dB（act 側再貢獻 −1.70 dB ≈ 32%）
   - act-only（fp16 權重 + act 量化全 W）：15.16 dB —— 證明 low-rank 分支
     的「原始 x」路徑已屏蔽主方向的 act 誤差；剩餘 act 誤差全落在 W_res。
2. **輸出增益收縮**（25 層 × 16 caches）：y_q ≈ k·y + n，k median 0.9988
   且 22/25 層 < 1；收縮主要來自 act 側（k_act ~0.996 vs k_wgt ~0.9995，
   attn2.to_out 低至 0.97）；能量比 ≈ 1（噪聲補位）→ FID 軟化的機制。
3. **smoothing 在 H-SVD residual 上的 act 側增益**（16 層，SVDQuant 官方
   smooth factors）：median +0.74 dB，**層型高度異質**——ffn_up 最高
   +3.55、to_q +0.75、out_proj/ffn_down ≈ 0（偶爾 −0.2）。
4. **clip 教訓**：weight QSNR 18.81→19.20（+0.4）但 MJHQ-2500 PSNR
   17.99→17.89（−0.10）——weight 側局部指標不可信；n=128 proxy 選細部
   權衡（clip）解析度不足，但選粗旋鈕（λ，兩次驗證一致）可靠。

---

## 計畫 C：摺疊式增益校正（Folded Gain Correction）— 主攻 FID

**動機**：數據 2 的乘性收縮（k<1）是 FID 紋理軟化的直接機制。文獻對應:
PTQD（NeurIPS 2023, arXiv 2305.10657）的 correlated noise correction——
他們報告 FID −0.48 / sFID −6.55；另見 QNCD（2403.19140）、
Timestep-Aware Correction（ECCV 2024）、Q-Drift（2603.18095）。

**做法**（比 PTQD 更輕：他們 runtime 除法，我們離線摺疊零成本）：
1. qdiff-128 caches 上 per-layer per-output-channel 估
   k̂_c = (Σ y_q,c·y_c + εE)/(Σ y_c² + εE)（正則化 + clamp [0.95, 1.05]）。
2. 把 1/k̂_c 摺進 wcscales（per-channel，kernel 原生）+ lora_up 行 +
   packed bias。注意 wcscales/bias 的 pack_perm_vector 排列。
3. 可選延伸：TAC 式 timestep-aware k（需 runtime per-step 向量乘，先不做）。

**預期**：FID 改善（PTQD 量級 −0.3~0.5），PSNR 幾乎不動。
**成本**：實作 ~2h、校準 pass ~20min、qdiff 選擇 ~20min、正式 2500 ~2.5h。

## 計畫 S：選擇性 smoothing — 攻 act 側（PSNR + FID 雙收）

**動機**：數據 3——smoothing 增益集中在 ffn_up/to_q；全面套用會在
out_proj/ffn_down 白付出（甚至 −0.2 dB）。先前 FLUX 全面 smoothing 有害
的舊結論（8/27）是在 PCA 基底 + 無 FID 判準下做的，不適用於現狀。

**做法**：
1. build 時 per-layer-type（或 per-layer，用 qdiff caches 上的 act 側
   SNR 增益門檻，如 >+0.3 dB）決定該層用 SVDQuant 官方 smooth factor
   還是 s=1。
2. 套 smooth 的層：W_res 在 smoothed 域量化（W_res·diag(s)、
   H_s = H/ssᵀ），lora 仍在原始 x 域（kernel 語意本來如此，
   lora_down 存 U/s——沿用 FLUX build_checkpoint 的既有機制）。
3. H-SVD basis 仍從原始 H 算（lora 吃原始 x）。
**預期**：act 側 MSE ×~0.84 → 端到端 +~0.2 dB PSNR；deadzone 軟化同步
緩解（FID 協同）。
**成本**：builder 加分支 ~2h、build+qdiff 選擇 ~1h、正式 2500 ~2.5h。

## 計畫 G：act-aware GPTQ — 耦合兩側誤差（低成本）

**動機**：GPTQ 目前用乾淨 x 的 H 校準，但部署時輸入是 Q(x/s)。把 H 換成
量化後輸入的統計，權重格點在「實際含噪輸入」下最優——等效於帶噪聲的
ridge 正則。相關：Timestep-Aware SVDQuant-GPTQ（2605.27003）同族思路。

**做法**：collect_cov 的 hook 改累積 H_q = E[Q(x/s) Q(x/s)ᵀ]（一次校準
pass），GPTQ/hsvd 用 H_q（或 hsvd 用原 H、GPTQ 用 H_q——兩種組合都排）。
**預期**：+0.05~0.15 dB（小但幾乎免費）。
**成本**：改動極小；重收 cov ~40min、build+選擇 ~1h。

## 計畫 R：act-aware block 輸出重建（AdaRound-style）— PSNR 主槓桿

**動機**:GPTQ 是逐層貪婪目標；文獻的 block 重建（BRECQ, 2102.05426;
AdaRound）直接最小化 block 輸出 MSE,W4 等級通常 +0.3~1.0 dB。diffusion
上的已知陷阱（PTQ4DiT, 2405.16005）:act 噪聲破壞 block 內依賴假設——
解法是把 act 量化放進重建 loop（act-aware）。與 SVDQuant 的關係:他們的
100-iter lowrank 最佳化也是校準期最佳化,我們只是目標更準（block 輸出、
含 act 噪聲）,協定對等。

**做法**：
1. 逐 transformer block:凍結格點 scale（two_level 現值）,把每層殘差的
   捨入決策鬆弛為 AdaRound 變數 h∈[0,1]（sigmoid + 正則退火）,可選一併
   微調 lora_up（fp16 自由參數,rank 不變）。
2. 目標:‖block_q(x) − block_fp(x)‖²,x 來自 qdiff-128 caches,forward
   內含 act_fp4_sim（act-aware）。Adam 幾百步/block。
3. 收斂後硬化捨入 → 同格點 codes → 打包,部署不變。
4. 順序可 sequential（前綴 block 用量化版產生輸入流,重用
   build_pixart_sequential.py 的 memmap 串流機制）。
**預期**：+0.3~1.0 dB PSNR（文獻量級）,是唯一可能把領先拉到 >0.3 dB 的
單一手段。
**成本**：實作 1~2 天;校準計算 ~28 block × 幾百步,估 2~4h GPU;
qdiff 選擇 + 正式 2500 照舊。

---

## 建議執行順序與組合

1. **S + G 合併一輪**（實作快、風險低）：qdiff-128 上排
   {baseline, S, G, S+G} 四組 → 勝者冻結上 2500。
2. **C**（FID 軌道,與 1 正交,可疊加在勝者上）。
3. **R**（大槓桿,最後做;若 1/2 已把 PixArt 拉開,R 可只做 FLUX 或略過）。

FLUX 側:任何在 PixArt 勝出的組合,同法移植(clip 的 FLUX 重審一併排入
該輪,見 PLAN.md 方向 3)。

## 參考文獻

- PTQD: arXiv 2305.10657 (NeurIPS 2023) — correlated/uncorrelated 噪聲分解
- QNCD: arXiv 2403.19140；TAC: ECCV 2024 (2407.03917)；Q-Drift: 2603.18095
- BRECQ: arXiv 2102.05426；AdaRound: arXiv 2004.10568
- PTQ4DiT: arXiv 2405.16005；EDA-DM: 2401.04585；QuEST: 2402.03666
- Timestep-Aware SVDQuant-GPTQ (Wan2.2): arXiv 2605.27003
- DMQ: ICCV 2025；CBQ: 2312.07950
