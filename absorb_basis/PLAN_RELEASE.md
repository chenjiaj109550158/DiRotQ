# PLAN_RELEASE：AbsorbQuant 公開 repo 整理（v1）

狀態：執行中（2026-09-04）。目標：`/home/dev/AbsorbQuant` 新 repo，
使用者僅憑該 repo 即可跑 ours，兩條路徑，且兩條路徑產物**位級一致**
（防造假疑慮）：

- **Path A（from-scratch）**：qdiff-128 caches → cov/act 統計 →
  selfsmooth 向量 →（凍結 config 或 select.py 重推導）→ build →
  與 release 權重逐 tensor 位級比對。
- **Path B（下載）**：從 HF 下載我們這幾週 quant 好的權重直接推論；
  `verify.py` 提供與 Path A 重建結果的位級比對工具。

## 定案六權重（HF 上傳清單）

| model | config | 檔案 | 大小 |
|---|---|---|---|
| pixart-sigma | damp0.1+S_rms@0.5 | pixart_selfsmooth_damp0.1_rms0.5.pt | 325M |
| sana-1.6b | damp0.3+S_rms@0.25 | sana_selfsmooth_damp0.3_rms0.25.pt | 822M |
| sdxl-turbo | damp0.3 | sdxl_r3_damp0.3.pt | 1.7G |
| sdxl-base | damp0.001 | sdxlb_damp0.001.pt | 1.7G |
| flux-schnell | damp0.01 | dirotq-selfsmooth-base-fp4_r32-flux.1-schnell.safetensors（改名） | 6.6G |
| flux-dev | damp0.3+S_rms@0.25 | fluxdev_selfsmooth_damp0.3_rms0.25.safetensors | 6.6G |
| (optional) pixart MX | damp0.3+S_rms@0.25 | pixart_mx_damp0.3_S0.25.pt | 308M |

合計 ~17.7G（+MX 308M）。全部為 PLAN_SELFSMOOTH 零依賴版定案。

## 新 repo 結構

```
AbsorbQuant/
  README.md  LICENSE(Apache-2.0)  NOTICE  pyproject.toml
  docs/{METHOD,THEORY}.md
  absorbquant/            # pip package
    paths.py gptq.py quant_utils.py mx_quant.py mx_kernel.py
    seeds.py              # per-filename seed 協定（vendored hash_str_to_int）
    families/{flux,pixart,sana,sdxl}/
      collect_calibration_dataset.py collect_cov*.py build_*.py
      selfsmooth_vectors*.py generate/runner
  configs/<model>.yaml    # 凍結：model_id、sampler、rank32、λ*、S 族+α、
                          # gptq damp、release 檔名+sha256
  datasets/               # qdiff-128 prompts、MJHQ metadata（含出處）
  scripts/
    calibrate.py --model X   # caches → cov/act → selfsmooth 向量
    build.py     --model X   # 凍結 config → final checkpoint
    select.py    --model X   # optional：Algorithm-1 全重推導（λ 網格+S 守門）
    generate.py  --model X --ckpt Y --prompt "..."
    verify.py    --model X --built A --ref B   # 逐 tensor 位級 + sha256
    download.py  --model X   # HF 下載 Path B
  baselines/README.md     # svdq 重現指引（deepcompressor optional）
```

## 遷移原則

1. **copy-and-rewire 不重寫**：位級一致性靠「同一份數值程式碼」保證，
   只改 import 與路徑常數（`/home/dev/DiRotQ`、DC 硬編碼 → paths.py）。
2. deepcompressor 完全退出必經路徑：runtime=nunchaku(pip)+Triton；
   eval 種子協定 vendored（附出處）；DC 只在 baselines/ optional。
3. flux 容器 template：預設抓我們自己的 HF release 當 layout 殼
   （--official 可換官方 nunchaku 殼），容器稽核保證所有校準衍生
   tensor 全數重建——兩種殼結果位級相同。
4. sdxl 缺自家 cache collector → 新增（照 pixart 版模式，
   eulera4-g0 / euler30-g5.0）。
5. 研究性 driver（round3/p23/headroom/lambdaext/rotation 遺產）不遷移，
   留在私有研究 repo 作 provenance。

## 位級一致性驗證（防造假核心）

- `verify.py`：packed int4/scale/lora 每個 tensor `torch.equal`；
  全檔 sha256；不等時報 per-tensor maxdiff。
- 決定性外殼（README 誠實聲明）：同 GPU 世代（SM120）+ 釘死
  torch/CUDA 版本 → 位級一致；異架構 → 數值等價（fp32 累加序），
  verify 改報容差統計。
- **本輪實測驗證**：pixart 在新 repo 走完整 Path A（caches 重生 →
  cov 重收 → selfsmooth → build）→ 與 release 檔位級比對。
  快檢先行：新 repo + 既有統計 build → 應完全 bit-equal（證明
  rewiring 未動數值）。

## 步驟

1. 骨架 + LICENSE/NOTICE/pyproject/configs ✍
2. 遷移 package 模組 + scripts 五件套 + sdxl collector
3. 快檢：既有統計 build pixart → bitwise vs release
4. 全 Path A pixart from-scratch（detached ~2h）→ bitwise vs release
5. README（兩路徑逐條指令）+ docs 遷移
6. HF 上傳清單與指令回報使用者
7. （後續）其餘五模型 Path A 復驗、select.py 選單重推導復驗

## 執行進度（2026-09-04）

- 遷移完成：29 檔 copy-and-rewire（43 處 import 改寫、殘留絕對路徑 0）、
  32/32 模組 import 乾淨。vendored：dc_mini（tree_map/hash_str_to_int/
  ceil_num_groups/pack_w4）、dc_patch（ConcatConv2d）、nunchaku_pack（原
  已自包含）。研究 driver 不遷移。
- **快檢通過**：新 repo + 既有統計重建 pixart → vs release
  `VERIFY_OK: 1568 entries bit-identical`（檔案 sha256 因 torch.save
  容器 metadata 而異 → verify.py 增加規範化 content digest）。
- **收集器忠實性（位級 vs 協定 caches）**：
  - pixart：prompt 0006 全 20 步×2 CFG，40/40 bit-identical。
  - sdxl：兩個協定細節被驗證揪出並內建——(1) DC 用 AutoPipeline 載
    fp32 轉 fp16（非 variant="fp16"，末位差）；(2) DC 校準前把 up-block
    resnet conv1 重寫成 ConcatConv2d（數學等價但 fp16 捨入不同 →
    denoise 軌跡不同）。修正後 4/4 bit-identical（SDXL_COLLECTOR_MATCH）。
  - sana/flux：sana 同構 transformer（重寫為 no-op）；flux 統計鏈本就
    出自我們收集器（vault 有 5120/5120 驗證備份）。
- pixart 全量 Path A（caches 重生→cov→vectors→build→verify）執行中。
- AbsorbQuant repo git 初始 commit `be2488c`。
