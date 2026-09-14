# 現況快照

## 主要目標

1. 整理 repository，讓外部開發者能快速開始開發，並發布到 AWF 共同 Git；備妥使用影像與模型權重的取得方式及下載連結。
2. 在這台電腦完整複現專案，備妥環境、影像、權重及必要中間資料，實際驗證流程。

推論資料流盤點是上述目標的前置工作。只以可確認的事實更新文件，未知項目保留為議題。

## 現況（2026-09-14）

- Push 前複查通過：Git 可納入的 119 個檔案約 668 KiB，涵蓋全部搬移，未混入資料／權重／輸出；Python 3.10 語法與模型雜湊通過。
- 乾淨候選副本驗證通過：移除既有 CARLA2REAL_*／PYTHONPATH 後，从 root/docs 執行檢查與初始化均成功；未 stage、commit 或 push。
- 完成：重新檢查分類方案；維持功能分類，先保留模型與資料布局。注意多支 Python 有 import 即執行行為，套件化不等於已可作為 API。
- 完成：更新 .gitignore，修正 datasets 排除失效、保留頂層資料說明與 docs/assets 圖片，排除權重／產物／環境／本機設定；23 個路徑案例驗證通過。
- 完成：依使用者授權搬移 60 個檔案至 carla2real/、configs/、scripts/、experiments/，同步更新 import、Shell 呼叫與使用文件；舊根目錄入口不保留 wrapper。
- 完成：docs/REORGANIZATION.md 與 FILE_MIGRATION.json 記錄完整對照、驗證與既有缺口；新增 scripts/check_layout.py 可重跑靜態檢查。
- 驗證：78 個 Python 語法、19 個 Bash 語法、60 檔搬移完整性通過；40 支 Python 執行 AST 與 18 支 Shell（排除預期引用變更）一致，模型核心雜湊一致。
- 驗證：從 root/docs 定位與初始化、環境覆寫、13 個 ignore 案例及文件連結通過；未跑 GPU／完整推論。
- 已確認：使用者指定「AWF 開源交付」與「本機完整複現」兩項主要目標。
- 完成：依使用者指示先建立預設資料／權重／輸出共用目錄；真實下載 URL 留待上傳後填入。
- 完成：docs/ASSET_DOWNLOADS.md 預留下載欄位，記錄放置位置、輸出命名與待填版本／校驗資訊。
- 完成：scripts/init_asset_dirs.py 可在新 clone 建立相同目錄，並可選擇場景／模型子目錄；預設模式重複執行成功。
- 完成：建立 Repository Guidelines（AGENTS.md）。
- 完成：靜態追蹤 recording、render 入口、Dataset、Generator、輸出、後處理與交付。
- 完成：新增 docs/INFERENCE_FLOW.md，含 Mermaid 圖、路徑表及 Q1–Q8。
- 使用者已同意建立 notes/progress.md 與 notes/history.md。
- 未執行 GPU 推論；checkpoints 與條件圖根目錄已建立但無真實資產，DVP 尚未提供。

## 活躍問題

| 議題 | 狀態 |
|---|---|
| AWF 發布位置與資產下載位置 | 下載清單已預留；使用者確認目前無真實連結，上傳後再填 |
| 本機複現範圍與環境 | 先從推論流程建立可驗證案例；是否包含重新訓練及本機硬體／執行環境待確認 |
| Q1 實際使用版本與權重 | 待提供執行命令／範例資料 |
| Q2 當前 65 類 label 前處理 | 舊程式只處理另一資料夾的 19 類 |
| Q3 v75 texture 與 make_v50r.sh | 程式碼未接通／缺檔 |
| Q4 DVP 合成輸出副檔名 | make_v33 寫 mp4，render 找 avi |
| Q5–Q8 前處理依賴、資料契約、路徑與外部感知 | 詳見流程文件；尚未修正 |
| 整理後仍缺少的 7 支 Shell 輔助程式 | check_layout.py 會明列；詳見 REORGANIZATION.md，非本次搬移造成 |

## 下一步

1. 待資產上傳後填入 docs/ASSET_DOWNLOADS.md；目前不以缺少 URL 阻擋目錄與文件準備。
2. 盤點本機硬體、執行環境與儲存需求；取得一組輸入及對應權重，先釐清 Q1–Q3 並跑通推論。
3. 使用新入口補齊缺失輔助程式與前後處理；實際資產到位後執行推論與比對，勿將靜態檢查當作完整複現。
4. 建立資產清單（來源、版本、授權、大小、校驗值、下載位置），備妥下載／安裝與驗證流程。
5. 驗證他人能依文件從乾淨環境開始，再完成 AWF 發布；訓練複現範圍另行確認。
