# DiRotQ-absorb-basis：完整方法定版（零 SVDQuant 依賴）

狀態：2026-09-02 定版。對應實驗記錄：PLAN.md（總表）、PLAN_ROUND2/3.md、
PLAN_SDXL30.md、PLAN_FLUXDEV.md、PLAN_SELFSMOOTH.md（去依賴消融）。
本文件描述「一個使用者要把 ours 部署到一個 diffusion transformer 上」
的完整流程；全程**不需要 SVDQuant 的任何校準結果**。

---

## 1. 方法核心：吸收基底分解 + 真 NVFP4 殘差量化

每個被量化的線性層 `y = xWᵀ`（W ∈ R^{oc×ic}）分解為：

```
y ≈ (x U_r)(W U_r)ᵀ  +  Q4(x/s) · Q4(W_res · diag(s))ᵀ
     └─ rank-32 lora ─┘   └────── W4A4 NVFP4 主分支 ──────┘
W_res = W − (W U_r) U_rᵀ（低秩殘差）
```

- **lora 分支吃 raw x**（fp16/bf16 GEMM），不經 smoothing——這是與
  SVDQuant 的關鍵結構差異：他們的 lora 在 smoothed 域，我們的
  smoothing（若啟用）只作用在 4-bit 主分支。
- s 為選擇性平滑因子（見 §4），未選層 s = 1。
- runtime 佈局與 SVDQuant/nunchaku 的 `SVDQW4A4Linear` 完全相同
  （qweight/wscales/smooth/lora_down/lora_up/wtscale），kernel 共用
  → 部署速度/記憶體 parity（實測 ±3%，見 PLAN.md 速度表）。

### 1.1 H-SVD 基底（取代 SVDQuant 的 plain SVD）

以校準二階矩 H = Σ x xᵀ 加權選基底：

```
C = chol(H + λ·mean(diag(H))·I)        # λ = 阻尼（Algorithm 1 自動選）
U_r = 前 r 個 eigenvector of Cᵀ (WC)ᵀ(WC) C⁻¹ 方向   # 實作：SVD_r(W C) 再 C⁻¹ 回拉
```

等價目標：min ‖(W − L)X‖ 的 rank-r 解（活化加權低秩逼近），λ 在
「活化加權 ↔ 純權重 SVD」間內插。r = 32 全程固定。

### 1.2 殘差量化：GPTQ on 真 NVFP4 兩級網格

W_res（或 W_res·s）用 GPTQ 量化，**量化網格 = kernel 實際解碼的
NVFP4 兩級結構**：per-tensor top scale（fp32 wtscale）×
per-group-16 e4m3 micro scale × e2m1 碼本。act-order（依 diag(H)
降序）內建；GPTQ 的 H 用：無 smooth 時即校準 H；有 smooth 時解析變換
`H/(s⊗s)`（**不另收 smoothed 域 cov**——這是去依賴要求之一）。
qkv 融合層用 per-out-channel top（wcscales，kind="qkv"）。
per-channel top 曾三模型消融皆負，永久排除（PLAN_ROUND3）。

### 1.3 12288 維 down-projection（FLUX `--down-absorb`）

SVDQuant 對 down_proj 量化但不 smooth；我們以同樣的 H-SVD+GPTQ 流程
重建（cov 以 keys-per-pass 分批收集，13 passes），取代其處理。

---

## 2. 校準資料協定（嚴格與 SVDQuant 對齊）

- prompts：**qdiff-128**（Q-Diffusion 的 128 條校準 prompts），
  與 SVDQuant 相同集合、相同數量。
- caches：各模型自帶 `collect_calibration_dataset.py`，以原始 fp16/bf16
  模型跑 128 prompts × 全步數，記錄 transformer 每步輸入
  （per-prompt 檔名 hash 種子 → 決定性可重生）。
- 所有 λ/S/α 選擇**只用** qdiff-128 生成圖對 bf16 ref 的四判準
  （§5）；官方測試集（MJHQ）從不參與選擇。

## 3. 校準統計收集（全部自有、串流、O(d²) 記憶體）

| 產物 | 腳本 | 用途 |
|---|---|---|
| cov 主表 H（每 hook d×d） | `collect_cov.py` / `collect_cov_{pixart,sana,sdxl}.py` | H-SVD、GPTQ、rms_x=√diag(H) |
| down cov（FLUX 12288²） | `collect_cov_down.py` | down 層 H-SVD/GPTQ |
| act samples（每 hook ~4096 列） | `collect_act_samples.py` | FLUX 離線增益量測 |
| act amax（每通道絕對最大） | `collect_act_amax.py` / hook pass | amax 族 s（消融用） |
| silu(temb) 樣本（FLUX） | `collect_temb_flux.py` | adanorm 活化加權重量化（§6） |

記憶體對比：SVDQuant 的 act 收集 O(樣本×token×d) 無 offload
（SDXL-base 1024px 實測 OOM）；我們全部串流累積 O(d²)。

## 4. 自足式選擇性平滑（S_closed，PLAN_SELFSMOOTH 定版）

### 4.1 閉式因子（rms 族，正式 pipeline 採用）

```
s_k = rms_{x,k} / rms_{w,k}
rms_{x,k} = sqrt(diag(H)_k)                    # cov 免費取得
rms_{w,k} = ‖W_res(λ*)[:,k]‖_rms（hook 內各層併排後取）  # 殘差感知
```

存全強度 s（幾何平均正規化為 1），builder 套 `s^α`。
**理論錨點**：α=0.5 時 s^{1/2} 是可分離上界
`‖ΔY‖² ≤ (Σ_k rms²_{x,k}/s²_k)(Σ_k rms²_{w,k} s²_k)` 的精確全域最小元
（AM-GM），且界針對的正是實際被量化的 W_res。
（amax 族 `amax_x/amax_w` 僅作消融對照，SANA 試點敗給 rms。）

### 4.2 逐層增益量測與守門層選擇

對每層以真活化算 `gain = 10·log10(e0/e1)`，
`e0=‖Q4(X)W_resᵀ−XW_resᵀ‖²`、`e1=‖Q4(X/s)(W_res·s)ᵀ−XW_resᵀ‖²`
（s 取 α=1）。FLUX 用離線 act samples
（`selfsmooth_vectors_flux.py`）；pixart/sana/sdxl 用 16 個 strided
cache 的真 forward hook（`selfsmooth_vectors_hook.py`；amax 全流累積
同 pass）。**只有 per-hook 平均 gain > τ=+0.3dB 的 hook 進入 smoothing**。

### 4.3 smoothing 範圍（與 SVDQuant 對齊）

不 smooth：FLUX down 層（12288）、SDXL conv、各模型 cross-attn KV、
所有未量化層。s 為 per-hook（pixart/sana/sdxl）或 per-fused-layer
（FLUX）。

## 5. Algorithm 1：校準期自動配置（全自動、僅校準資料）

四判準：PSNR / LPIPS / SSIM / FID-proxy-128（clean 特徵高斯 FID），
候選 qdiff-128 圖 vs bf16 ref qdiff-128 圖。

```
Stage A（λ）：λ ∈ {0.001,0.003,0.01,0.03,0.1,0.3} 各 build+gen，
              兩兩對決積分（wins 加總）最高者 = λ*
Stage B（S）：對 α ∈ {0.25,0.5,0.75,1.0} 依序 build(λ*, S_closed, α)+gen；
              對「現任最佳」贏 ≥3/4 判準才接受（貪婪、no-regret：
              全拒即回退純 λ*，不會比基底差）
官方測試：僅最終配置改變時重跑（MJHQ，五指標，vs SVDQuant 官方
checkpoint/我們代跑其 pipeline 的 checkpoint）
```

單配置邊際成本：build 5–11 分（小模型）/ ~30 分（FLUX 12B）+
qdiff-128 生成 5–20 分。

## 6. FLUX 容器與 adanorm 自足重量化

FLUX 部署走 nunchaku C++ loader，以官方 safetensors 僅作**佈局容器**：

- 所有量化層 tensors（qweight/wscales/wcscales/smooth/smooth_orig/
  lora_down/lora_up/wtscale）由我們的 build 全數覆寫。
- **76 層 adaLN modulation linears（W4A16 int4 g64）也自足重量化**
  （稽核發現的隱藏依賴）：`collect_temb_flux.py` 用 qdiff-128 prompts
  的 CLIP-L pooled × timestep 網格造 silu(temb) 樣本（~2 分鐘），
  以通道能量 diag 加權的非對稱 (min,max) 收縮網格搜尋量化，打包與
  deepcompressor 轉換器逐位對拍。實測輸出誤差比官方 AWQ 好 15×。
- **容器稽核**（`selfsmooth_driver.audit_flux_container`）：build 後
  逐 key 驗證所有校準後綴 tensor 與官方容器不同（全 1 smooth 與單元素
  純量巧合豁免），未過即中止。
- 剩餘與官方相同的內容 = BFL 原始 bf16 權重與檔案佈局（非校準產物）。

pixart/sana/sdxl 不經容器：自建 kernel .pt + 原始 fp16/bf16 模型
python 注入（`run_*_kernel_generate.py`），adaLN 保持原精度。

## 7. 各模型特殊層（與 SVDQuant skip 對齊）與最終配置

| 模型 | 特殊層處理 | 零依賴定案 | 官方戰績 vs SVDQuant |
|---|---|---|---|
| FLUX-schnell | adanorm W4A16 自足重量化；qkv wcscales；single proj_out 拆分；down 不 smooth；embedder/最終層 bf16 | λ0.01 | **5:0**（MJHQ-1000） |
| FLUX.1-dev | 同上 | λ0.3 + S_rms@0.25 | **5:0**（MJHQ-500） |
| PixArt-Σ | cross-attn KV fp16；adaln_single/caption_proj/proj_out fp16 | λ0.1 + S_rms@0.5 | **5:0**（MJHQ-2500） |
| SANA-1.6B | cross-attn KV bf16；conv_depth bf16；GLUMBConv 1×1 當 linear；pad-first 對齊 | λ0.3 + S_rms@0.25 | **4:1**（MJHQ-2500，僅 FID-ref） |
| SDXL-Turbo | cross-attn KV fp16；34 resnet conv im2col 量化 | λ0.3 | **5:0**（MJHQ-2500） |
| SDXL-base 30 步 | 同上 | λ0.001 | **5:0**（MJHQ-1000） |

**合計 29/30 指標勝，全零依賴。**

## 8. 依賴稽核總結（使用者部署接觸面）

需要：目標模型原始權重、qdiff-128 prompts、我們的 collectors/builders、
deepcompressor 開源框架（基礎設施）、nunchaku kernel（parity 主張刻意
共用）、FLUX 官方檔僅作佈局容器（稽核保證無校準位元殘留）。
**禁用且已驗證不觸碰**：SVDQuant smoothing dump（smooth.pt/scale.pt/
model.pt）、官方 checkpoint smooth 解包、svdq-s 域 cov
（cov_actq_smooth）、官方 adanorm 量化位元。

## 9. 校準成本（實測，RTX 5090）

| 成分 | 成本 |
|---|---|
| caches（兩法共同前置） | 5m40s（pixart）–2h04m（dev 50 步） |
| cov 串流收集 | 小模型 1–2h；dev 4h59m |
| 閉式 s + 增益量測 | **3.5–13 分/模型**（+FLUX temb 2 分） |
| Algorithm-1 選單（λ 6 點 + S×α 4 點） | 小模型 ~1.5–3.5h；dev ~7.5h |
| 對照：SVDQuant 校準 | 5–7h/模型產 1 配置（smoothing gridsearch 為大宗） |

同等 wall-time 下我們產 10–14 配置的自動搜索與守門證據；借用因子
路線的 5–7h 前置被 ~10 分鐘閉式計算取代。

## 10. 腳本地圖（repo 路徑 = absorb_basis/）

- 分解與 build：`build_checkpoint.py`（FLUX，含 `--down-absorb`、
  `--select-smooth-vectors/-gains/-alpha`、`--adanorm-temb`）、
  `pixart/build_pixart_kernel.py`、`sana/build_sana_kernel.py`、
  `sdxl/build_sdxl_kernel.py`（皆吃 `--smooth-pt {skey:s}` + `--gains`）
- 收集：`collect_cov*.py`、`collect_act_{samples,amax}.py`、
  `collect_temb_flux.py`、各模型 `collect_calibration_dataset.py`
- 自足 smoothing：`selfsmooth_vectors_flux.py`、
  `selfsmooth_vectors_hook.py`
- 選擇與評測：`selfsmooth_driver.py`（四判準守門 + 容器稽核 + 官方）、
  `round3_driver.py`（λ/α 歷史定版）、各模型 `run_*_kernel_generate.py`、
  `flux_gen_nunchaku.py`
- 結果：`results/{model}_selfsmooth_{qdiff128,test*}.json`、
  `results/speed_table.json`
