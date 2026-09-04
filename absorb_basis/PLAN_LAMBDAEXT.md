# λ 網格上端擴展：turbo / dev 重跑（Algorithm-1 grid top-end）

狀態：**執行完畢，配置採納待使用者裁決**（2026-09-03 02:39 全鏈完成；
執行結果與決策選項見文末）。動機：svd-basis 消融
（THEORY.md 實驗閉環）顯示兩個 λ\* 頂到網格上緣（0.3）的模型，最優
點在網格之外——turbo 1:3、dev 0:4 輸給 λ→∞ 端點。其餘四模型
λ\* 對 ∞ 全勝（3×4:0、1×3:1），不在本輪範圍。

## 設計

- **新網格點**：λ ∈ {1, 10, 1e6}（1e6 ≈ ∞ 端點；其 qdiff 圖已在
  svd-basis 消融生成，直接沿用）。
- **Stage A′ 排名**：候選 = {現任 λ\* 基底} ∪ {1, 10, 1e6}，
  qdiff-128 四判準 pairwise 積分。原網格下緣五點在 round3 已於同協定
  輸給 0.3，不重排（引用）。dev 側全部候選皆為 temb-adanorm build
  （零依賴版一致性；svdinf 圖即是）。
- **Stage C**：若 λ\* 更新 → 以新 λ\* 重算閉式向量+增益
  （s 依賴 W_res(λ\*)），S_rms × {0.25,0.5,0.75,1.0} 貪婪守門。
- **官方**：最終配置 ≠ 現行零依賴配置才重跑（turbo MJHQ-2500、
  dev MJHQ-500），vs SVDQuant + 併記前任零依賴配置作三方對照。
- 全程零 SVDQuant 校準產物；dev build 稽核照常。

## 預估

| 步驟 | turbo | dev |
|---|---|---|
| λ1、λ10 build+gen（1e6 圖快取） | ~35m | ~1h50m |
| 排名（含快取圖統計） | ~15m | ~20m |
| 新 λ\* 向量+增益 | ~12m | ~4m |
| S 選單 4α | ~1h10m | ~3h40m |
| 官方（配置變更時） | ~1h15m | ~1h35m |
| 合計 | ~3.5h | ~7.5h |

產物：`results/{sdxl,fluxdev}_lambdaext_qdiff128.json`、
`{...}_lambdaext_test{2500,500}.json`；chain
`run_lambdaext.sh`，log `results/lambdaext_chain.log`。
磁碟 237G 起跑（dev cov_down 43G 已預留在本地）。

## 執行結果（2026-09-02 20:48 – 09-03 02:39，~5.9h GPU）

### Stage A′ 排名（qdiff-128 pairwise 積分）

- turbo：**λ\*=1e6**（積分 10，對 0.3/1/10 全勝）；曲線：像素三項隨
  λ↑ 單調改善、FID-proxy 在 0.3 最佳（判準內部分歧 3:1）。
- dev：**λ\*=1e6**（λ10 對 0.3 全勝、1e6 再勝 λ10）。

### Stage C（新基底上的 S_rms 守門）

- turbo：四點全拒（一貫）→ 定案 **damp1e6**。
- dev：0.25/0.5 拒、**0.75 過門 4:0**、1.0 拒 → 定案
  **damp1e6+S_rms@0.75**（α 甜蜜點隨基底從 0.25 移到 0.75——
  SVD 基底殘差更「白」，耐受更強的 smoothing）。

### 官方結果（vs SVDQuant；併記前任零依賴配置）

**turbo MJHQ-2500**（`sdxl_lambdaext_test2500.json`）：damp1e6 =
19.375/0.2099/0.6918/**11.832**/35.245 → 對 SVDQuant **4:1**
（FID-GT 35.245 vs 35.206 以 0.04 落敗；其餘四項均大於前任
damp0.3 的邊際，FID-ref −0.30）。前任 damp0.3 為 5:0。

**dev MJHQ-500**（`fluxdev_lambdaext_test500.json`）：
damp1e6+S_rms@0.75 = 21.425/0.1983/0.8130/**36.380**/94.578 →
對 SVDQuant **5:0** 維持。與前任（damp0.3+S_rms@0.25，同為 5:0）
頭對頭 1:4：FID-ref 大勝（−1.45）、其餘四項毫釐小輸。
qdiff 上 +0.43dB 的優勢未轉移（官方 PSNR 反而 −0.04）。

### 科學結論

1. **λ 極端處出現校準→官方泛化縫隙**：turbo 的 qdiff FID-proxy
   反對票忠實預告了官方 FID-GT 翻車；dev 的 qdiff 全票通過卻只兌現
   FID-ref。qdiff-128 四判準在 λ→∞ 端解析度受限。
2. α 甜蜜點隨基底移動（0.25→0.75），佐證「S 強度須隨殘差結構重選」
   ——Algorithm-1 的 α 網格設計再獲支持。
3. 兩配置對 SVDQuant 都是穩定勝方；差別只在我們自己的記分板細節。

### 決策選項（待裁決）

- **A（凍結原網格）**：turbo/dev 維持前任零依賴配置（總表 29/30）；
  本輪作為 grid-sensitivity ablation 呈現（含泛化縫隙觀察，係
  FID-proxy 判準價值的正面證據）。
- **B（採納擴展網格）**：turbo→damp1e6（4:1）、dev→damp1e6+S_rms@0.75
  （5:0）；總表 28/30，但 turbo 四項與 dev FID-ref 邊際更大，
  且與「校準期全自動選擇」協定完全一致。
- 不可取：依官方結果逐模型挑選（測試集選擇=協定違規）。

## 裁定（2026-09-04）

使用者裁定 **A：凍結原網格 {0.001…0.3}**。主表 29/30 不變；本輪
（turbo/dev λ→1e6）以消融呈現：校準端一致偏好 1e6、官方端 turbo
FID-GT 毫釐翻車（4:1）/ dev 互比 4:1 偏 λ0.3——「校準→官方泛化縫隙
＋qdiff-128 解析度下限」的第四案例；1e6=λ→∞（plain SVD）端點實測。
release configs（AbsorbQuant）不變（turbo/dev λ=0.3 已為凍結值）。
