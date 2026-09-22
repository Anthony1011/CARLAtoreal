# 影像、權重與輸出位置

更新日期：2026-09-14。所有專案資產下載點目前均為 **待上傳**。下列欄位是發布準備，不代表資產已提供或已驗證；上傳後直接將「待填」替換為下載連結。

## 建立資料夾

在 repository 根目錄執行（Windows 可用 `python`，Linux 可用 `python3`）：

```bash
python scripts/init_asset_dirs.py
```

此命令只建立不存在的資料夾，不下載資料、不建立假權重、不覆蓋檔案。空資料夾不會隨 Git 保存，因此新 clone 後需執行一次。本次僅建立共用根目錄；場景、model、tag 未確定前不預先填入。

```text
datasets/
└── training_v12_mapillary/       推論條件圖
pix2pixHD/
├── checkpoints/                 模型權重
└── results/mp4/
    ├── vision_pilot/            render 的影片產物
    └── NEW/                     scripts/delivery/refresh_new.sh 整理的交付索引
output/                          最終交付影片
├── vp_input_1024/               外部感知用縮小影片
├── calibrated/                 外部感知輸出影片
└── gt/                         評分用 GT JSON（不是模型輸入）
```

確認要使用的場景後可建立對應子目錄，例如：

```bash
python scripts/init_asset_dirs.py --town Town05 --weather sunny
```

這是命名範例，不表示 Town05 是已選定的驗證資料。它建立 `datasets/recorded_Town05_sunny_inst/{rgb,semantic}/`，以及 `datasets/training_v12_mapillary/test_Town05_sunny_inst_gt_{label,edge,depth,normal,chroma,label_rich}/`。夜間改用 `--weather night`，chroma 改為 light。`label_rich` 供後處理使用。

權重目錄用 `--model MODEL_NAME` 建立；MODEL_NAME 須換成已確認的真實模型名稱。上述選項可合併使用。instance 目錄由 recorder 在 `--instance` 啟用時建立；texture 支援尚未接通，見推論流程 Q3。

本工具刻意對應目前 render 的 repository 預設路徑，不讀取自訂 CARLA2REAL_DATA／CARLA2REAL_OUT。若要使用外部磁碟，須先處理 render 硬編 `ROOT/datasets` 的落差。

## 下載清單

所有位置相對於 repository 根目錄。`NAME`、`PHS`、`MODEL` 是格式變數，不是要建立的字面資料夾名稱。

| 資產 | 下載連結（上傳後替換） | 放置位置／內容 | 版本、大小、SHA-256 |
|---|---|---|---|
| CARLA 錄製影像包 | 待填 | `datasets/recorded_NAME/`；內含 `rgb/`、`semantic/`、`frame_speed.txt`，有錄製 instance 時才附 `instance/` | 待確認 |
| 預先產生的推論條件圖包 | 待填 | `datasets/training_v12_mapillary/PHS_*`；label、edge、depth、normal、chroma（日）或 light（夜），及後處理 label_rich | 待確認 |
| 晴天 Generator 權重 | 待填 | `pix2pixHD/checkpoints/MODEL/latest_net_G.pth` | MODEL、版本與相容旗標待確認 |
| 夜間 Generator 權重 | 待填 | `pix2pixHD/checkpoints/MODEL/latest_net_G.pth` | MODEL、版本與相容旗標待確認 |
| 範例輸出影片（供比對） | 待填 | `output/`；應附產生時的命令、模型與場景資訊 | 待確認 |
| 感知評分 GT（若提供） | 待填 | `output/gt/<town小寫>_<weather>_gt.json` | 待確認 |

`NAME` 例如 `Town05_sunny_inst`；`PHS` 對應 `test_Town05_sunny_inst_gt`。模型讀的是逐幀影像與條件圖資料夾；單獨一支 MP4 不能取代它們。目前沒有已確認的 MP4 → 完整條件圖流程。

README 宣告 sunny v75／night v76，但權重與本 checkout 的相容性尚未驗證，所以清單不將它們標成可直接使用的下載版本。訓練影像來源另見 [datasets/README.md](../datasets/README.md)，本次未建立訓練資料布局。

## 輸出命名

- 模型逐幀結果：`pix2pixHD/results/MODEL/PHS_EPOCH/images/*_synthesized_image.jpg`。
- render 影片：`pix2pixHD/results/mp4/vision_pilot/<town>/<town>_<weather>_<TAG>_FINAL_1920.mp4`。
- 交付影片：`output/<town>_<weather>_vp55_<TAG>_FINAL_1920_visionpilot.mp4`，1920×960；檔名中的 visionpilot 不代表已加 HUD。
- 推論 log：`pix2pixHD/checkpoints/render_<TAG>_log.txt`。
- 外部感知 log／評分：`output/logs_<TAG>/`，由執行流程建立。

模型、epoch、tag 的動態結果目錄由程式執行時建立，無需建立空的假結果。

## 上傳後填寫

每個下載包填入實際 URL，並附檔名、版本、大小、SHA-256、解壓縮目錄層級、來源與授權說明。權重須列明對應程式版本及推論旗標；影像和條件圖須說明對應場景及幀命名。從新 clone 實際驗證下載、解壓及推論後再標示「已驗證」。既有缺口見 [INFERENCE_FLOW.md](INFERENCE_FLOW.md)，資產條款整理見 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
