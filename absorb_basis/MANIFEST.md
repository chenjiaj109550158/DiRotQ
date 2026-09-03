# MANIFEST：重要檔案分類清單（2026-09-04 整理）

慣例：`$M` = `/home/dev/DiRotQ/models`、`$DC` =
`/home/dev/deepcompressor/examples/diffusion`、`$V` =
`/vault/dirotq-absorb-backup`。**vault 皆有副本**（backup.sh 持續同步）；
「重生成本」= 本機 RTX 5090 從頭重做的 GPU 時間。

## 1. SVDQuant 權重（基線側，比較用；非我們的產物）

| 檔案 | 大小 | 說明 |
|---|---|---|
| `$M/flux-dev/svdq-fp4_r32-flux.1-dev.safetensors` | 7.0G | 官方 nunchaku dev NVFP4（下載件；也是我們 build 的佈局容器） |
| HF cache `mit-han-lab/nunchaku-flux.1-schnell` | 6.6G | 官方 schnell NVFP4（同上雙用途） |
| `$M/pixart-sigma/absorb_basis/pixart_svdq_kernel.pt` | 0.34G | 我們代跑其 pipeline 後轉出的基線 kernel（重生=其完整校準 5–7h） |
| `$M/sana-1.6b/absorb_basis/sana_svdq_kernel.pt` | 0.86G | 同上 |
| `$M/sdxl-turbo/absorb_basis/sdxl_svdq_kernel.pt` | 1.78G | 同上 |
| `$M/sdxl-base/absorb_basis/sdxlb_svdq_kernel.pt` | 1.78G | 同上（其校準 7h10m） |

（其 smoothing dump `svdq_model_dump/` 已全數移往 vault-only——
pipeline 禁用，僅供歷史對照。）

## 2. Ours 權重（零依賴定版配置 = 發表配置，部署即用）

| 模型 | 檔案 | 大小 | 配置 |
|---|---|---|---|
| FLUX-schnell | `dirotq-selfsmooth-base-fp4_r32-flux.1-schnell.safetensors` | 7.0G | λ0.01（無 S） |
| FLUX.1-dev | `fluxdev_selfsmooth_damp0.3_rms0.25.safetensors` | 7.0G | λ0.3+S_rms@0.25 |
| （dev 選單基底） | `fluxdev_selfsmooth_base.safetensors` | 7.0G | λ0.3 無 S（重排名/消融用） |
| PixArt-Σ | `pixart_selfsmooth_damp0.1_rms0.5.pt` | 0.34G | λ0.1+S_rms@0.5 |
| （pixart 基底） | `pixart_absorb_damp0.1_kernel.pt` | 0.34G | λ0.1 |
| SANA | `sana_selfsmooth_damp0.3_rms0.25.pt` | 0.86G | λ0.3+S_rms@0.25 |
| （sana 基底） | `sana_r3_damp0.3.pt` | 0.86G | λ0.3 |
| SDXL-Turbo | `sdxl_r3_damp0.3.pt` | 1.77G | λ0.3（基底=定版） |
| SDXL-base | `sdxlb_damp0.001.pt` | 1.77G | λ0.001（基底=定版） |

重生成本：FLUX 系 build 各 ~30 分（factor/adanorm cache 命中 ~20 分）；
小模型 5–15 分——前提是 §5 的校準統計在手。

**λext 懸決候選（PLAN_LAMBDAEXT 選項 B；選 A 即可刪）**：
`fluxdev_lext_damp1e6{,_rms0.75}.safetensors`（14G）、
`sdxl_lext_damp1e6.pt`（1.8G）。

## 3. bf16/fp16 Reference 圖（**最貴的重生項**——官方五指標的分母）

| 路徑 | 張數 | 大小 | 重生成本 |
|---|---|---|---|
| `$DC/runs/fluxdev-ref/samples/MJHQ/MJHQ-500` | 500 | 630M | **~5h42m**（bf16 50 步 cpu-offload） |
| `$DC/baselines/torch.bfloat16/flux.1-schnell/.../MJHQ/MJHQ-1000` + `YAML/qdiff-128` | 1128 | 1.5G | ~2h |
| `$DC/baselines/torch.float16/pixart-sigma/dpm20-g4.5/.../MJHQ-2500` | 2500 | 3.4G | ~1h |
| `$DC/runs/sana-ref/samples/MJHQ/MJHQ-2500` | 2500 | 3.8G | ~1h |
| `$DC/runs/sdxl-ref/samples/MJHQ/MJHQ-2500` | 2500 | 948M | ~1.5h |
| `$DC/runs/sdxlb-ref/samples/MJHQ/MJHQ-1000` | 1000 | 1.5G | **~2.5h**（30 步） |
| 各模型 `*-qdiff-ref`（qdiff-128 bf16 ref，守門分母） | 128×6 | ~1G | 每模型 20m–1h22m |

## 4. GT 基準（MJHQ-30K 子集，FID-GT 分母）

`$DC/benchmarks/MJHQ-GT-{500,1000,2500}`（0.5/1.0/2.5G）——由已刪的
HF MJHQ-30K 資料集抽出；重生 = 重新下載 3.4G + 檔名對映。

## 5. 校準統計（ours 側全部產物；重生成本見括號）

| 檔案 | 大小 | 重生 |
|---|---|---|
| `$M/{flux-schnell,flux-dev}/basis/absorb_cov_basis.pt` | 6.8G×2 | cov 收集 ~1–2h（需 caches） |
| `$M/*/basis/absorb_cov_{pixart,sana}.pt`、`absorb_cov_sdxl_{linear,conv}.pt` | 9–22G/模型 | 同上 |
| `absorb_act_samples{,_down}.pt`、`absorb_act_amax.pt`、`adanorm_temb.pt`、`noise_diag.pt` | 各 0.002–7.7G | 分鐘級–25m |
| `selfsmooth_{rms,amax}{,_down}*.pt` + gains json（各模型） | KB 級 | 3.5–13m/模型 |
| `$M/flux-*/basis/factor_cache/`、`absorb_basis/adanorm_cache.pt` | ~9G/模型 | 首次 build 自動重建 |
| **vault-only**：flux 兩模型 `absorb_cov_down`（43G×2）、qdiff-128 caches（全模型） | — | cov-down ~2.5h；caches 5m–2h（種子決定性） |

## 6. 官方測試圖（ours 與 svdq 各配置；重生 = 對應 kernel 生成）

`$DC/runs/{flux,fluxdev,pixart,sana,sdxlb}-selfsmooth-final`（零依賴官方
圖，0.6–3.8G 各）、`*-svdq-final*` 與 `nvfp4-nunchaku-flux.1-schnell`
（SVDQuant 官方圖）、`*-p23-final`/`*-lext-final`（消融官方圖）。
重生 24m–1.5h/組。

## 7. qdiff-128 證據圖庫（所有歷史候選的守門記錄）

`$DC/runs/*qdiff*`——每目錄 128 張、數十 MB；是排名可重算性的依據
（bootstrap CI 的原料），全數保留。

## 8. HF 模型快取（可重下載，斷網才是問題）

六個原始模型 + 兩個 nunchaku 官方檔，共 ~150G
（`~/.cache/huggingface/hub`）。

---
狀態：本輪清理後磁碟 **222G 可用**；已刪（vault 有）：被否決的
candidate 權重 39G、flux cov_down 86G、qdiff caches ~22G。
結果 JSON 全在 git（`absorb_basis/results/`）。
