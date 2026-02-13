"""各チャンクに priority メタデータを一括設定するスクリプト"""
import json, os

DATA_DIR = "backend/data"

# priority: 3=名物・代表的（必ず上位に）, 2=主要, 1=一般
PRIORITY_MAP = {
    # === 草津温泉 ===
    "kusatsu_001": 1,  # アクセス情報
    "kusatsu_002": 3,  # 湯畑（草津のシンボル）
    "kusatsu_003": 3,  # 湯もみと踊り（名物）
    "kusatsu_004": 2,  # 西の河原公園
    "kusatsu_005": 3,  # 草津三湯（代表的温泉施設）
    "kusatsu_006": 1,  # 温泉門・町内交通
    "kusatsu_007": 2,  # スキー場
    "kusatsu_008": 3,  # 歴史と泉質（日本三名泉）
    "kusatsu_009": 3,  # 独自の温泉文化（湯もみ・合わせ湯）
    "kusatsu_010": 1,  # 問い合わせ先
    "kusatsu_011": 2,  # 冬季イベント
    "kusatsu_012": 1,  # 冬季注意事項
    "kusatsu_013": 2,  # 観光施設
    "kusatsu_014": 2,  # おすすめの楽しみ方
    "kusatsu_015": 3,  # 効能とおすすめの人

    # === 有馬温泉 ===
    "arima_001": 3,  # 金泉と銀泉（有馬の代名詞）
    "arima_002": 2,  # 公衆浴場
    "arima_003": 2,  # 泉源めぐり
    "arima_004": 2,  # お土産・名産品
    "arima_005": 1,  # 飲食・グルメ
    "arima_006": 2,  # 観光名所
    "arima_007": 1,  # アクセス
    "arima_008": 2,  # 温泉街散策
    "arima_009": 1,  # 宿泊施設
    "arima_010": 2,  # おすすめの楽しみ方
    "arima_011": 3,  # 泉質と効能
    "arima_012": 3,  # 効能とおすすめの人

    # === 別府温泉 ===
    "beppu_001": 3,  # 概要と歴史（源泉数日本一）
    "beppu_002": 2,  # 別府八湯（前半）
    "beppu_003": 2,  # 別府八湯（後半）
    "beppu_004": 2,  # 主要観光スポット
    "beppu_005": 3,  # 地獄めぐり（別府の代名詞）
    "beppu_006": 2,  # 名物グルメ
    "beppu_007": 1,  # アクセス
    "beppu_008": 3,  # 泉質の多様性（10種類）
    "beppu_009": 2,  # おすすめの楽しみ方
    "beppu_010": 3,  # 効能とおすすめの人

    # === 箱根温泉 ===
    "hakone_001": 2,  # 概要
    "hakone_002": 3,  # 箱根十七湯（代表的特徴）
    "hakone_003": 3,  # 箱根登山電車・ロープウェイ（名物）
    "hakone_004": 3,  # 美術館（彫刻の森等、有名）
    "hakone_005": 3,  # 大涌谷・芦ノ湖（代表的景勝地）
    "hakone_006": 2,  # 神社仏閣
    "hakone_007": 2,  # ハイキング
    "hakone_008": 1,  # 体験スポット
    "hakone_009": 2,  # 四季のイベント
    "hakone_010": 1,  # アクセス
    "hakone_011": 1,  # ペット同伴
    "hakone_012": 3,  # 泉質と効能
    "hakone_013": 2,  # おすすめの楽しみ方
    "hakone_014": 3,  # 効能とおすすめの人

    # === 温泉基礎知識 ===
    "onsen_knowledge_001": 3,  # 温泉の定義と泉質（基礎中の基礎）
    "onsen_knowledge_002": 3,  # 草津温泉の概要
    "onsen_knowledge_003": 1,  # アクセス
    "onsen_knowledge_004": 2,  # 観光スポット
    "onsen_knowledge_005": 2,  # 草津三湯
    "onsen_knowledge_006": 1,  # 冬季注意
    "onsen_knowledge_007": 1,  # 問い合わせ先
    "onsen_knowledge_008": 2,  # 箱根概要
    "onsen_knowledge_009": 2,  # 別府・有馬概要
    "onsen_knowledge_010": 2,  # 日帰り入浴
    "onsen_knowledge_011": 2,  # 基本マナー
    "onsen_knowledge_012": 2,  # 季節の楽しみ方
    "onsen_knowledge_013": 3,  # 泉質別おすすめの人
}

for fname in os.listdir(DATA_DIR):
    if not fname.endswith("_chunks.json"):
        continue
    path = os.path.join(DATA_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    changed = 0
    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        priority = PRIORITY_MAP.get(cid, 1)
        chunk["metadata"]["priority"] = priority
        changed += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"{fname}: {changed} chunks updated")

print("Done!")
