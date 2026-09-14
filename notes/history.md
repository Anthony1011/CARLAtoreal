# 決策與歷程

## 2026-09-14：建立推論流程盤點

使用者要求依輸入來源、處理程式、模型入口、輸出位置理解推論，並明確要求不猜測。採靜態程式碼追蹤；README 的 baseline 宣告不當成實際部署證據。完整結果及可追溯議題存於 docs/INFERENCE_FLOW.md。

關鍵發現：README 宣告的 v75 texture 與晴天 make_v50r.sh 不在目前入口實作中；prepare_gt_test_label.py 是 19 類舊流程，不能直接接到 render 的 65 類輸入；make_v33.py 寫 mp4，但 render 尋找 avi。這些是盤點結果，尚未執行或修正。

使用者同意新增 notes 記錄。保留 docs/STATE.md 和 docs/EXPERIMENTS.md 原內容，避免把本次理解流程的工作混入既有模型實驗歷史。

## 2026-09-14：確認專案兩項交付目標

使用者明確指定：一、整理為方便他人開發的開源專案，發布到 AWF 共同 Git，並備妥影像與權重下載連結；二、在目前電腦建立所需影像、權重與環境，完整複現。推論流程盤點因此定位為交付前置工作，而非最終目標。

後續以本機實際驗證支持開源文件與流程。AWF repository、資產來源／託管位置，以及是否包含從頭重新訓練尚未確定；不假定缺少的權重已可取得或發布。

## 2026-09-14：先建立目錄與下載占位文件

使用者指定目前沒有實際下載連結，先按程式建立影片、權重與輸出位置，上傳後再貼入 URL。新增 docs/ASSET_DOWNLOADS.md，所有下載欄位明確標為待填，不建立假連結或資產。

共用目錄依 render_model.sh 的預設 repository 布局建立。因空目錄不會隨 Git 保存，新增只用標準函式庫的 scripts/init_asset_dirs.py，預設建立共用目錄，提供 --town/--weather/--model 供確認後建立子目錄；不推測實際模型或場景。本機已重複執行預設模式成功，未執行模型或下載。

## 2026-09-14：分類方案複核與 Git 排除規則

功能分類方案可作為第一階段整理，但 Python 腳本多有頂層讀檔／執行副作用，不能以全面 import 驗證，也不能把移入套件視為完成 API 整理。維持 pix2pixHD 與既有資料位置；版本配方只是另行分類，不視為可刪除。使用者本次僅要求複核及 ignore，未搬移程式。

既有 datasets/ 被 !datasets/ 重新納入，非影像副檔名的資料可能漏入 Git。改為 /datasets/* 並僅保留頂層 Markdown；另保留 docs/assets 下的小型文件圖片。補齊環境、模型、壓縮包及本機設定排除，不全域排除 JSON/TXT，以免隱藏設定或清單。git check-ignore --no-index 驗證 11 個排除路徑與 12 個保留路徑全部通過；diff 空白檢查通過。

## 2026-09-14：依確認方案執行目錄整理

使用者授權開始整理。60 個根目錄 Python/Shell 檔案依功能搬入 carla2real/、configs/、scripts/、experiments/；完整對照保存在 docs/FILE_MIGRATION.json。保留 pix2pixHD 模型核心及資料／權重／產物位置，未改 dependency 或演算法。舊 CLI 檔案位置由新模組命令／腳本路徑取代，不保留 wrapper；版本配方保留，make_v33 仍屬主鏈後處理。

Python 改用 carla2real.config 與 carla2real.common.vidcodec；Shell 改用 python -m，包含版本配方的內嵌 Python import。configs/config.sh 由自身上層取得 ROOT 並匯出 PYTHONPATH，維持 driver 切換工作目錄後仍可找到本專案套件。加入 .gitattributes 固定 Shell 為 LF，以支援 Windows checkout 後的 Bash 執行。歷史文件只加搬移導覽，不改原實驗紀錄。

完整驗證結果見 docs/REORGANIZATION.md。語法、引用、檔案完整性、模型雜湊、演算法 AST 與路徑 smoke checks 通過；第一次 AST 比對遇到 Windows 預設 cp950 解碼問題，改以 UTF-8 讀取 Git 原文後通過。已有 7 支缺失輔助程式仍留為議題；沒有假造替代功能。未執行訓練、GPU 推論或資料清理流程。

## 2026-09-14：Push 前再次驗證

按 Git tracked + untracked（排除 ignore）且仍存在的檔案建立 119 檔候選副本，不變更 index。確認所有 60 個新位置均包含，沒有 bulk 資產、單檔超過 5 MiB 或模型核心變更；Python 語法另外按 3.10 檢查。以候選內容建立乾淨暫存副本，清除 CARLA2REAL_*／PYTHONPATH 後，从 root/docs 執行 check_layout 與 init_asset_dirs 均通過，python -m carla2real.config 也通過。

本次未發現新的整理缺陷。7 支既有輔助程式缺失與未進行完整推論的限制仍在；此版本適合作為目錄整理提交，尚非已驗證的完整模型發布。暫存區為空，尚未 stage、commit 或 push。
