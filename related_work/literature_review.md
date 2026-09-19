# Related Works 逐篇精讀筆記

研究問題：在控制 architecture、parameter count、optimizer 等變數後，BatchNorm 與 residual connection 對淺層（6–26 層）CNN optimization 與 generalization 的獨立貢獻是什麼？這些貢獻是否隨 network depth 與 batch size 系統性改變？

---

## 第一組：ResNet 原始論文與 Identity Mappings

### 1. Deep Residual Learning for Image Recognition (He et al., 2016, CVPR)

**核心方法／論點**：出發點是 **degradation problem**，不是梯度消失——論文明確排除梯度消失假說（BN 存在下前向/後向訊號都健康），轉而猜測是「收斂速度指數級變慢」，但坦承機制未明。ImageNet：18 vs 34 層（plain 34 層變差，ResNet 34 層變好，深度效應反轉）。CIFAR-10：n={3,5,7,9} 對應 20/32/44/56 層，plain 與 residual 完全同構（identity shortcut、同參數量），是相對乾淨的單變數對照。

**啟發與發現**：
- 關鍵句：「18 層時 plain 與 residual 準確率相近，但 ResNet 收斂更快——SGD 在『不算太深』時仍能找到好解，residual 提供的主要是早期收斂速度」——幾乎是 H3 的原始表述。
- **plain net baseline 本身已含 BN**：整篇論文比較的都是「BN+無 residual」vs.「BN+residual」，BN=OFF 象限從未被測試——BN 沒有讓 degradation 消失，只是把上限推遲到 56+ 層。
- CIFAR-10 完整深度曲線（20→1202 層）是「residual 邊際效益隨深度變化」的雛形，但從未計算 Δ 值，也未取樣 20 層以下。

**不足**：只排除法、無機制解釋；CIFAR 深度點多數僅跑一次、無變異量報告；未探索 20 層以下；plain baseline 本身含 BN，BN 獨立貢獻未知；ImageNet 深度比較混淆了 block 類型（bottleneck vs. basic）。

### 2. Identity Mappings in Deep Residual Networks (He et al., 2016, ECCV)

**核心方法／論點**：數學證明恆等 skip + 恆等 activation 讓訊號可在任意兩 block 間直接傳播。消融證明任何非恆等 shortcut（scaling/gating/1×1 conv/dropout）都讓 ResNet-110 變差，即使新設計參數更多——用來論證「degradation 是優化問題而非表徵能力問題」。提出 pre-activation 設計，深度越深（1001 層）改善越大，較淺（110 層）改善很小。全部實驗最淺 110 層。

**啟發與發現**：
- 論證模式值得模仿：「拿掉某元件變差 ≠ 該元件表徵能力有用，要先排除優化問題」——呼應我們 H1a/H1b 要分開驗證。
- BN 放置位置的陷阱：BN 放在 addition 之後（而非分支內部）會讓效果變差（訊號被 BN 干擾）——提醒我們 Experiment A 的 BN 必須放在每個 conv 分支內部，不能放錯位置。
- 同一改動「深層效果放大、淺層幾乎沒差」的模式，與 H3 假設的交互作用同構，可作旁證。
- 統計方法示範：5 次獨立訓練報告中位數/mean±std——這是社群既有標準，不是我們憑空要求的嚴謹度。

**不足**：完全未測淺層（最淺 110 層）；channel 改變處 identity 理論本身不嚴格成立（作者自承），對應我們 notebook 裡用 1×1 projection 的地方需要小心；「優化容易度」與「BN 正則化效果」混在一起講、未分開量化；未探索 batch size；未與 ensemble/singularity 假說對話。

---

## 第二組：BatchNorm 機制論文

### 3. How Does Batch Normalization Help Optimization? (Santurkar et al., 2018, NeurIPS)

**核心方法／論點**：用「Noisy BatchNorm」（故意注入噪音破壞分布穩定性）反駁 internal covariate shift 假說——訓練表現幾乎不受影響。改用嚴謹量測（gradient predictiveness、effective β-smoothness）證明 BN 真正效果是讓 loss landscape 更平滑：BN 網路的 gradient predictiveness 比無 BN 網路好近兩個數量級。VGG on CIFAR-10：有 BN 83% vs 無 BN 80%（無增強）。

**啟發與發現**：
- 提供 H1a 最直接的量化證據與可沿用的三種診斷指標（loss 沿梯度方向變化、gradient predictiveness、effective β-smoothness），比單純 gradient norm 更精確。
- **關鍵坦承**（footnote 4）：「我們選擇不用 ResNet 做實驗，因為 ResNet 似乎提供類似 BN 的好處，會引入混淆變數」——這篇奠基性論文的作者親口承認自己迴避了我們現在要解決的問題。
- L1/L2/L∞ 等其他 normalization 也有類似平滑效果——BN 不是唯一能平滑地景的方法，寫作時不應把「平滑地景」講成 BN 專屬。

**不足**：完全未測 residual 架構；無深度掃描；VGG 主要結果似為單次訓練、無 mean±SD；只在 CIFAR-10 與合成任務驗證。

### 4. Batch Normalization Biases Residual Blocks Towards the Identity Function (De & Smith, 2020, NeurIPS)

**核心方法／論點**：理論證明無 BN 的 residual block，activation 變異數隨深度指數爆炸（residual/skip 各半）；有 BN 時只線性成長，residual branch 被迫只貢獻 1/(ℓ+1) 變異數比例，block 趨近恆等映射。驗證性實驗「SkipInit」（只用可學習純量 α，無任何 normalization）可訓練 1000 層，94.3% vs BN 94.6%，幾乎打平。Batch size 掃描發現：小 batch 時 BN/SkipInit/無 normalization 三者最佳學習率其實相近（推翻 Santurkar「BN 主要靠允許更大學習率」的說法——兩篇論文結論在此直接矛盾），但 BN 測試準確率仍顯著較高（額外正則化效果）。

**啟發與發現**：
- 部分回答 H4，但比預期精細：不是「小 batch 時 BN 優勢消失」，而是「大學習率優勢消失，但可訓練深度優勢與正則化優勢仍在」——H4 實驗要把「最佳學習率」與「測試準確率」分開報告。
- SkipInit 可作為 Experiment A 的第五對照組，把「normalization」與「downscale 效果」進一步拆開。
- **關鍵發現**：全篇所有對照組（BN／SkipInit／無 normalization）都保留 residual，從未測試 BN=ON、Residual=OFF——比 Santurkar 那篇的迴避更進一步證實我們研究的必要性（連這篇專門研究 BN×residual 交互作用的論文也只做了 2×2 factorial 的一半）。

**不足**：理論只嚴格證明前向傳播，反向靠引用類比；正則化效果解釋不完整（作者自承需靠額外 Dropout+調參才追上 BN）；從未測真正 plain（無 residual）架構；理論以「單層 residual branch」簡化推導，與實際多卷積層 block 有落差。

---

## 第三組：去 BN、留 Residual（我們研究的鏡像方向）

### 5. Fixup Initialization: Residual Learning Without Normalization (Zhang et al., 2019, ICLR)

**核心方法／論點**：理論證明標準初始化+residual+無 BN 會使梯度指數爆炸；提出三規則初始化（分支末層歸零、按 L^{-1/(2m-2)} 縮放、加純量 bias/multiplier）。可訓練 10,000 層（僅測 1 epoch）；ResNet-110/CIFAR-10：7.24%（BN 6.61%、Xavier-init 7.78%）；加 Mixup/Cutout 後可追平甚至超過 BN。

**啟發與發現**：
- Appendix C.1（Fig 4）有全部 Related_Works 中**唯一**「完全移除 residual」的直接消融實驗！但只在 110 層做、只看前 3 epoch 的 minibatch training accuracy，無 test accuracy、無多 seed——證明訓練動態不同，但無法回答最終準確率差距，也未觸及淺層。
- Theorem 2 給出的初始化縮放公式可直接用來設計 Experiment A 的 BN-OFF 組，排除「初始化沒調好」這個混淆變數。
- 重要提醒：拿掉 BN 後即使優化完全穩定，測試誤差仍有差距（額外正則化效果）——(h) 討論須把「optimization 差距」與「generalization 差距」分開處理。

**不足**：所有深度實驗聚焦「極深」（10+ 到 10,000 層），未觸及 6–26 層淺層區間；深度曲線只測第一個 epoch，反映初始化品質而非最終收斂；唯一的 residual 移除消融样本量、深度、指標都不足；未測 CIFAR-100。

### 6. A Robust Initialization of Residual Blocks...without Batch Normalization (Civitelli et al., 2022)

**核心方法／論點**：發現 Brock et al. (NFNets) 的初始化只保留前向 variance、反向梯度 variance 隨深度指數爆炸（ResNet-50/101 訓練直接失敗，但 ResNet-18 成功）。提出同時保留前向+反向 variance 的方案（IdShort/LearnScalar/ConvShort）。ConvShort（每 block 都用完整 1×1 conv shortcut，參數量與 BN 版打平：38.02M）才能真正追平 BN；IdShort/LearnScalar 參數量少 38%（23.47M）但有落差。3 次獨立實驗報告 mean±SD。

**啟發與發現**：
- 現成的「深度造成質變」實證案例：同一初始化在 R18 能用、R50/101 失敗——支持我們「尋找深度轉折點」的研究設計。
- **具體數字證實 projection shortcut 混淆參數量的問題**：同一骨幹只因 shortcut 用 identity 還是 1×1 conv，參數量差 38%，且只有參數量較多者追平 BN——這是「residual 優勢可能只是 shortcut 額外容量」的直接證據，強化我們 Experiment A 必須控制/報告參數量的必要性。
- 再次印證 BN 的正則化效應：不用 BN 可訓練到 SOTA，但需要更長訓練時間與更強增強來補償正則化流失。

**不足**：最淺架構是 ResNet-18，且只在附錄附帶測試；從未測試「同時拿掉 BN 又拿掉 residual」的純 Plain CNN；只 3 次重複、無正式統計檢定；三種方案本質是比較「不同程度的 skip 表達能力」，非乾淨的 residual on/off 二元對照。

---

## 第四組：Residual 機制的競爭假說（集成 / Loss Surface）

### 7. Residual Networks Behave Like Ensembles of Relatively Shallow Networks (Veit et al., 2016)

**核心方法／論點**：把 residual network 展開成 2^n 條路徑的集合。病灶研究：刪除單一 block，VGG 立刻崩潰、110 層 ResNet 幾乎不受影響；打亂 block 順序，誤差隨相關係數平滑上升（ensemble 特徵）。路徑長度服從二項分布（mean=27），梯度隨路徑長度指數衰減，有效梯度集中在 5–17 層長的路徑（僅占 0.45%）；只用有效路徑長度重訓練，準確率無顯著差異（5.96% vs 6.10%）。

**啟發與發現**：
- 提供 H3 的機制性支持方向：若「有效路徑長度」只有 5–17 層，當名目深度本身就落在此範圍內（如我們的 6–26 層），residual「壓縮超深網路成淺路徑集合」的好處可能無用武之地——直接呼應 Section 4「6–10 層時 residual 幾乎沒差」的觀察。
- 但 Veit et al. 從未在淺層（如 6→54 層掃描）重複這個路徑長度/有效梯度分析，只在 110/200 層測過——這正是我們 Experiment B 要填的空缺，且可直接借用「單一 block 刪除病灶測試」方法套用到我們的架構上。

**不足**：作者自承「深度仍是開放研究問題」，非決定性結論；residual block 全部內建 BN，從未拆解 BN 角色；只在 110/200 層驗證，未掃描深度；重訓練實驗僅一次、無多 seed；ensemble 假說與 degradation/梯度消失假說並非互斥，論文未裁決孰為主導。

### 8. The Loss Surface of Residual Networks: Ensembles and the Role of Batch Normalization (Littwin & Wolf, 2016/2017)

**核心方法／論點**：用 spherical spin-glass 模型把 ResNet loss 表示成不同路徑長度的加權和，權重由純量 β=ρnC/√Λ 控制。理論證明：訓練初期有效集成集中在淺路徑，隨訓練進行 C（BN 情境下即 BN 縮放參數 λ_l）單調增加，把有效深度逐漸推深——即「有效深度」是動態成長過程，不是靜態值。10 層合成任務實驗觀察到 λ_l 隨訓練增長（Fig 1d）；即使無 BN，權重 L2 norm 也有類似增長模式（Fig 1e）。

**啟發與發現**：
- 把「有效深度」從靜態升級成動態，且驅動機制正是 BN 的縮放參數——為 H1a 提供更具體的候選機制：BN 可能主動驅動「訓練中逐步解鎖更深有效路徑」的過程，而非只是讓地景平滑。
- 推論：若總深度本身很淺（如我們 6–24 層），這個動態成長空間天花板很低，很快觸頂——不論有無 residual，BN 或單純權重尺度增長本身可能已足夠讓網路用到「幾乎全部」深度，使 residual 額外的路徑集合優勢被壓縮到近乎零。
- 論文唯一的真實驗證用的正是 10 層網路（落在我們深度範圍內！），但沒有做「無 BN」或「無 residual」的對照比較——我們的 Experiment B 若在深度 10 附近記錄 λ_l（或無 BN 時的權重 norm）隨訓練增長曲線，可直接跟這篇 Fig 1(d) 對話。

**不足**：理論建立在作者自承「不成立」的簡化假設上（A1–A4，尤其輸入獨立性明顯違反現實）；唯一驗證是合成高斯分類玩具任務，非影像/CNN，與宣稱解釋的「極深網路」場景有巨大驗證斷層；只證明成長「方向」，未證明「終點」（穩態值如何依賴深度 p），這正是我們深度掃描可補上的空缺；未把 λ 成長軌跡與實際準確率提升串起來對照；本篇是 ICLR 2017 投稿版，正式發表版本內容是否有調整需另行確認。

---

## 第五組：Re-parameterization 與 Residual 形式本身

### 9. RepVGG: Making VGG-style ConvNets Great Again (Ding et al., 2021, CVPR)

**核心方法／論點**：訓練期多分支（3×3 conv + 1×1 conv + identity，各自接 BN）、推論期用代數等價把三分支摺疊成單一 3×3 conv。ImageNet 上首次讓純 plain 架構跑到 80%+ top-1。Ablation（RepVGG-B0，約 28 層）：無分支 72.39% → 只 identity 74.79% → 只 1×1 73.15% → 兩者皆有 75.14%（BN 恆存在於所有配置）。BN 位置實驗：拿掉 identity 分支的 BN 掉到 74.18%；BN 移到相加之後掉到 73.52%。

**啟發與發現**：
- 直接支持「residual 效益主要在訓練期、推論期架構本身不需要它」的核心角度。
- Table 6 的 ablation 相當於「BN 恆為 ON」情境下 residual 分支的邊際貢獻：在約 28 層、ImageNet-1k（1000 類）情境下是 +2.75pp，並非零——與我們 CIFAR-10、6–10 層測出「近乎零」形成對照，顯示資料集複雜度與深度都可能是交互作用的自變數（**更正**：先前曾把「CIFAR-10 vs. CIFAR-100/Tiny-ImageNet 敏感度差異」歸給 arXiv:2510.24036，查證原文後並無此內容；該句實出自 DeepAFL〔Tang et al., ICLR 2026，解析式聯邦學習〕，不可當作 CNN 證據引用）。
- **新發現的混淆變數**：BN 的「位置」（分支內部 vs. 相加後）本身造成 1.6pp 差距——我們 Experiment A 目前只切換「BN 有/無」，未控制「BN 相對 residual addition 的位置」，需要加入設計考量。
- Trivial Re-param／ACB 對照組顯示 residual 的效益不是單純過參數化（ACB 參數更多但效果更差）。

**不足**：作者自承 RepVGG 是為 GPU/專用晶片速度設計，較不關心參數量，用它支撐「邊緣/SRAM 受限」敘事需謹慎（驗證的是 GPU 推論速度，非我們原本關心的記憶體受限場景）；完全未測 CIFAR 規模或淺層網路，ablation 只在 28 層做過一次、未跨深度重複；所有配置皆含 BN，從未做 BN=OFF 對照。

### 10. Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks (Oh et al., 2025, NeurIPS)

**核心方法／論點**：質疑標準相加式 residual 本身，把模組輸出正交分解為平行分量（與現有 stream 同方向、視為冗餘）與正交分量（真正新方向），只保留正交分量更新。ViT 上效益顯著（ViT-B/ImageNet-1k: 73.27%→77.05%），但 **ResNetV2 上效益「明顯較小」**（作者原話 "more modest"）。切換連接類型實驗顯示效益主要發生在訓練「早期」。

**啟發與發現**：
- 本身就是「連怎麼加 residual 都還是開放問題」的第一手證據，2025 年最新 NeurIPS 論文仍在爭論這件事。
- **關鍵細節**（Table 2）：CIFAR-10 上 ResNetV2-18/34（較淺）正交更新有小幅提升，但 **ResNetV2-50/101（較深）幾乎打平甚至微幅退步**——即使是完全不同角度的介入（怎麼加，而非加不加），在「CIFAR-10 + 深層」組合上也出現效益消失甚至反轉，與我們「CIFAR-10 對 residual 相關介入不敏感」的觀察方向一致，可作旁證，但需加註但書（50/101 已換成 bottleneck block，與 18/34 的 basic block 不同，可能是 block 類型混淆而非純深度效應）。
- Switching-connection-type 方法論（用 optimizer state 保留/重置分開「架構改變」與「訓練被打斷」的效果）值得借鏡，用於檢查我們自己實驗的混淆變數。

**不足**：作者自承受算力限制未測更深 ResNetV2 或 web-scale 資料集，機制的深層理論解釋仍開放；完全未做 BN on/off 對照；深度網格一樣是官方標準深度組（18→34→50→101），50/101 換了 block 類型，非乾淨深度掃描；最淺測試對象仍是 ResNetV2-18，比我們關注的 6–10 層深得多。

---

## 綜合結論：文獻缺口總表

| 缺口 | 涉及論文 | 我們的因應 |
|---|---|---|
| 沒有人在 20 層以下、且有重複實驗（多 seed）的情況下測過 BN=ON、Residual=OFF（ResNet 原論文的 CIFAR-10 plain vs. residual 已是參數量對等且含 BN，但最淺 20 層、僅 ResNet-110 重複） | 全部 12 篇 | Experiment A/B 核心貢獻 |
| 深度比較多用「官方標準深度組」，混淆 block 類型等變數 | ResNet（ImageNet 部分）, Identity Mappings, Orthogonal Residual | Experiment B 用完全同構架構做密集深度掃描 |
| Residual 效益的統計嚴謹度普遍不足（單次訓練、無 mean±SD） | ResNet CIFAR 曲線多數深度點、RepVGG ablation、Orthogonal Residual 部分實驗 | 5+ seed、報告 mean±SD 與信賴區間 |
| Projection shortcut 混淆參數量 | Identity Mappings（channel 改變處）、Robust Init（38% 參數量差距實證） | 控制 in=out channel 或明確報告參數量 |
| BN 位置本身是自由度，不只是有無 | RepVGG Table 7 | Experiment A 需控制 BN 相對 addition 的位置 |
| Optimization 差距與正則化/generalization 差距常被混為一談 | Fixup、Robust Init、Identity Mappings | H1a（機制）與 H1b（量級）分開驗證 |
| Batch size 對 BN 的影響從未與 residual on/off 交互測試 | Group Normalization（僅測 BN vs GN，未測 residual）、De & Smith（僅在 residual 恆存在下測 batch size） | H4 需自行設計 BN×Residual×BatchSize 交互實驗 |
