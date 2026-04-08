"""
サポートレター作成支援ツール - ローカルサーバー v41
起動方法: py -m uvicorn server:app --reload --port 8000
ブラウザで http://localhost:8000 を開く
"""
import json
import os
import re
import random
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import uvicorn

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# System Prompts
# ============================================================

JOB_HISTORY_PROMPT = """あなたは米国ビザ申請のCV（職務経歴書）に記載する「職歴」を生成するAIです。

## 役割
申請者が入力した各職歴エントリーの情報をもとに、CVに使える職歴の文章を生成する。

## 前提
- 職歴は「業務経験」の具体版。業務経験が「見出し」なら、職歴はその「中身」
- 具体的な製品名・設備名・数値・成果を入れて、この人が何を達成したかを示す
- 過去形で記述する

## スタイル
- 2〜4文程度
- 「〜を担当し、〜を実現しました」のような具体的な実績記述
- 数値的な成果があればできるだけ含める

良い例：
「冷凍餃子の新規製造ラインの立ち上げに参画し、充填機および急速凍結装置の仕様選定とレイアウト設計を担当。試運転時に発生した包餡工程での生地破れを、温度管理条件の見直しにより解消し、予定通りの量産開始を実現した。」

## ルール
- 最初のメッセージから即座に文章を生成する。質問・挨拶・前置きは一切不要
- 各職歴エントリーを ===JOB_N_START=== と ===JOB_N_END=== で囲んで出力する（Nはエントリー番号）
- 入力された情報を忠実に反映する
- ユーザーからのフィードバックがあれば反映する

## 出力フォーマット（エントリー数に応じて）
===JOB_1_START===
（職歴1の文章）
===JOB_1_END===

===JOB_2_START===
（職歴2の文章）
===JOB_2_END===

（以下、エントリー数分続く）

生成後「この内容でよければ確定してください。修正があればお伝えください。」と伝える。"""

BUSINESS_OVERVIEW_PROMPT = """あなたは米国ビザ申請のサポートレターに記載する「米国子会社のビジネス概要」を生成するAIです。

## 役割
申請者が入力した米国子会社の情報をもとに、サポートレターに使えるビジネス概要の文章を生成する。

## スタイル
- 「当社米国子会社は主に、〜を行っています。」で始める
- 3〜5文程度のパラグラフ
- フォーマルなビジネス文書のトーン
- 業界の専門用語は使ってよいが、過度に技術的にならないこと

## ルール
- 最初のメッセージから即座に文章を生成する。質問・挨拶・前置きは一切不要
- ===BIZ_START=== と ===BIZ_END=== で囲んで出力する
- 入力された情報を忠実に反映する
- ユーザーからのフィードバックがあれば反映する

## 出力フォーマット
===BIZ_START===
（ビジネス概要の文章）
===BIZ_END===

生成後「この内容でよければ次に進んでください。修正があればお伝えください。」と伝える。"""

SITUATION_CHANGE_PROMPT = """あなたは米国ビザ申請のサポートレターに記載する「状況の変化」を生成するAIです。

## 役割
申請者が入力した変化と影響の情報をもとに、「なぜ今、人を派遣する必要があるのか」の背景となる状況変化の文章を生成する。

## 重要：状況の変化と課題の書き分け
この文章の前には「ビジネス概要」が、後には「課題」が別パートとして記載される。

「状況の変化」パートに書いてよいのは：
- 何が起きているか（事実）
- それによって何が影響を受けているか（影響）

「状況の変化」パートに書いてはいけないのは：
- 「〜が必要です」「〜が求められています」「〜が急務です」
- 「〜を強化する必要がある」「〜の改修が必要な状況」「〜の構築が求められる」
- 「対応しきれない」のあとに続く「〜が必要」「〜が求められる」系の表現
- 対策・対応・解決策に言及する表現すべて
→ これらは次の「課題」パートに書く

※ ユーザーの入力に「〜が急務」「〜が必要」「〜に対応しなければならない」等が含まれていても、そのまま使わないこと。事実と影響の部分だけを抽出し、対応・対策に関する部分は省略する。

良い例：
「リモートワークの普及により、企業の採用プロセスが急速にデジタル化しています。オンライン面接やバーチャル職場体験ツールへの需要が急増しており、従来型のシステムでは技術的に対応しきれない状況が生じています。」

悪い例：
「〜新たな技術要件と運用体制の構築が求められる状況が生じています」← 「構築が求められる」は課題
「〜の大幅な改修とアップグレードが必要な状況となっています」← 「必要な状況」は課題

## スタイル
- 「〜しています」「〜が生じています」「〜が進んでいます」のような現状描写のみ
- 2〜4文程度のパラグラフ
- 客観的・事実ベースのトーン

## ルール
- 最初のメッセージから即座に文章を生成する。質問・挨拶・前置きは一切不要
- ===CHANGE_START=== と ===CHANGE_END=== で囲んで出力する
- 入力された情報を忠実に反映する
- ユーザーからのフィードバックがあれば反映する

## 出力フォーマット
===CHANGE_START===
（状況の変化の文章）
===CHANGE_END===

生成後「この内容でよければ次に進んでください。修正があればお伝えください。」と伝える。"""

CHALLENGE_PROMPT = """あなたは米国ビザ申請のサポートレターに記載する「課題」を生成するAIです。

## 役割
申請者が入力した情報をもとに、「米国側でどんな課題を解決する必要があるか」の文章を生成する。

## 重要
- この文章の前には「ビジネス概要」と「状況の変化」が別パートとして記載される
- そのため、状況の変化の説明を繰り返さないこと
- 「課題」パートでは、その状況から生じる具体的な課題・不足だけを書く

## スタイル
- 前のパートを受けて「このような状況の中、〜が必要です」のように始めてもよい
- ただし状況自体の説明は1文以内にとどめ、すぐに課題の記述に入ること
- 「〜する必要があります」「〜が求められます」のような表現
- 2〜4文程度のパラグラフ
- 現地だけでは解決できない理由が含まれること

## ルール
- 最初のメッセージから即座に文章を生成する。質問・挨拶・前置きは一切不要
- ===CHALLENGE_START=== と ===CHALLENGE_END=== で囲んで出力する
- 入力された情報を忠実に反映する
- ユーザーからのフィードバックがあれば反映する

## 出力フォーマット
===CHALLENGE_START===
（課題の文章）
===CHALLENGE_END===

生成後「この内容でよければ次に進んでください。修正があればお伝えください。」と伝える。"""

REQUIRED_PERSONNEL_PROMPT = """あなたは米国ビザ申請のサポートレターに記載する「必要な人材」を生成するAIです。

## 役割
申請者が入力した情報をもとに、「だからこういう人材を日本から派遣する必要がある」という結論の文章を生成する。

## 重要
- この文章の前には「ビジネス概要」「状況の変化」「課題」が別パートとして記載される
- そのため、前のパートの内容を繰り返さないこと
- 「必要な人材」パートでは、どんな人材が必要で、なぜ日本から派遣するのかだけを書く

## スタイル
- 「現状課題に対応するため、〜の経験を持つ人材が必要です。」で始める
- 2〜3文程度のパラグラフ
- 最後は「以上のことから当ミッションを遂行する人材を日本から派遣いたします。」で締める
- 派遣の正当性が伝わること

## ルール
- 最初のメッセージから即座に文章を生成する。質問・挨拶・前置きは一切不要
- ===PERSONNEL_START=== と ===PERSONNEL_END=== で囲んで出力する
- 入力された情報を忠実に反映する
- ユーザーからのフィードバックがあれば反映する

## 出力フォーマット
===PERSONNEL_START===
（必要な人材の文章）
===PERSONNEL_END===

生成後「この内容でよければ次に進んでください。修正があればお伝えください。」と伝える。"""

DUTY_SYSTEM_PROMPT = """あなたは米国ビザ申請のサポートレターに記載する「米国での業務内容」の候補を生成するAIです。

## 役割
申請者の基本情報から、業務内容の候補を5つずつ生成する。

## 前提
- 申請者はこれから米国に赴任する人として扱う
- 業務内容は申請上の重要度が高くないため、一般的な内容で構わない
- ただし、その職種が実際に時間を使う業務を反映すること

## ルール
- 最初のメッセージから候補5つを即生成する。質問・挨拶・前置きは一切不要
- 業界の専門用語・略語は避け、専門知識がなくても内容が伝わる表現を使う
- 部門・ポジションにとって自然な用語を使う
- 「対象」として提供される製品名・設備名・システム名は生成の参考情報として使うが、業務内容の文面に毎回入れる必要はない。自然な場合のみ使うこと
- 過去に出した候補と同じ・類似の内容は出さない
- ユーザーから方向性のフィードバックがあれば反映する

## 出力フォーマット

===DUTIES_START===
1. [業務内容1]
2. [業務内容2]
3. [業務内容3]
4. [業務内容4]
5. [業務内容5]
===DUTIES_END===

各業務内容：
- 1〜2文、80〜200文字程度
- 「[業務対象]の[アクション]を行い、[目的・成果]」の構造
- ポジションに応じてマネジメント寄り／実務寄りに書き分ける
- フィードバック内容があれば反映する

生成後「この中にピンとくるものがなければ、追加で5つ生成できます。方向性を伝えていただければそれも反映します。」と伝える。

## 差し替えリクエストへの対応
- ユーザーから「N件を差し替えたい」というリクエストが来た場合、ちょうどN件だけ生成すること
- 5件ではなくN件を ===DUTIES_START=== ～ ===DUTIES_END=== で囲んで出力する
- 差し替え対象として提示された候補とは異なる方向性の候補を生成すること"""

EXP_SYSTEM_PROMPT = """あなたは米国ビザ申請のサポートレターに記載する「これまでの業務経験」の候補を生成するAIです。

## 役割
申請者の基本情報と、米国での業務内容をもとに、これまでの業務経験の候補を5つずつ生成する。

## 前提
- 業務経験は「この人が米国での業務を遂行するのに十分なスキル・知識を持っている」ことを示すためのもの
- 業務経験は職歴（CV）の「見出し」にあたる抽象度で書く。職歴にはより具体的な状況・数値・成果が書かれるが、業務経験ではそれらを抽象化して「どんな種類の経験をしてきたか」を示す
- 提供された「米国での業務内容」を遂行するために必要なスキル・知識を推測し、それに関連する過去の経験を生成すること

## 重要：業務経験の抽象度

業務経験は、具体的な製品名・数値・固有名詞を避けて、経験の種類を示す。

良い例：
- 食品製造ラインの新規立ち上げにおける設備選定およびレイアウト設計を担当。
- 製造工程における品質課題の原因分析と製造条件の最適化を実施。
- 海外工場への設備導入プロジェクトに参画し、現地エンジニアへの技術指導を担当。

悪い例（具体的すぎる。これは職歴のスタイル）：
- 冷凍餃子の新規製造ラインの立ち上げに参画し、充填機および急速凍結装置の仕様選定とレイアウト設計を担当。
- タイ工場向けの即席麺ラインの設備導入を担当し、現地エンジニアへの操作手順の指導を実施。

## ルール
- 最初のメッセージから候補5つを即生成する。質問・挨拶・前置きは一切不要
- 業界の専門用語・略語は避け、専門知識がなくても内容が伝わる表現を使う
- 具体的な製品名・設備名・国名・数値は入れず、一般的な表現に抽象化すること
- 数値目標の達成（売上XX%増加、コストXX%削減など）は入れない
- 過去に出した候補と同じ・類似の内容は出さない
- ユーザーから方向性のフィードバックがあれば反映する

## 出力フォーマット

===EXP_START===
1. [業務経験1]
2. [業務経験2]
3. [業務経験3]
4. [業務経験4]
5. [業務経験5]
===EXP_END===

各業務経験：
- 1文、簡潔に
- 過去形で記述する（「〜を実施」「〜を推進」「〜を担当」など）
- ポジションに応じてマネジメント寄り／実務寄りに書き分ける
- 経験年数に応じた深さ・幅の経験にする
- フィードバック内容があれば反映する

生成後「この中にピンとくるものがなければ、追加で5つ生成できます。方向性を伝えていただければそれも反映します。」と伝える。"""

# ============================================================
# Prompt mapping
# ============================================================

PROMPTS = {
    "job_history": JOB_HISTORY_PROMPT,
    "business_overview": BUSINESS_OVERVIEW_PROMPT,
    "situation_change": SITUATION_CHANGE_PROMPT,
    "challenge": CHALLENGE_PROMPT,
    "required_personnel": REQUIRED_PERSONNEL_PROMPT,
    "duty": DUTY_SYSTEM_PROMPT,
    "experience": EXP_SYSTEM_PROMPT,
}

# ============================================================
# API
# ============================================================

class ChatRequest(BaseModel):
    messages: list[dict]
    mode: str = "duty"


def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY が .env に設定されていません")
    return anthropic.Anthropic(api_key=api_key)


def clean_mission(text):
    """ミッションから数値・期間表現を除去する"""
    text = re.sub(r'[0-9０-９]+[%％]', '', text)
    text = re.sub(r'[0-9０-９]+ヶ月[以内にで]*', '', text)
    text = re.sub(r'[0-9０-９]+年[以内にで]*', '', text)
    text = re.sub(r'[0-9０-９]+倍[にへ]*', '', text)
    text = re.sub(r'年間', '', text)
    text = re.sub(r'[0-9０-９]+', '', text)
    text = re.sub(r'[ 　]+', '', text)
    text = re.sub(r'をを', 'を', text)
    text = re.sub(r'のの', 'の', text)
    return text.strip()


@app.get("/api/test-data")
async def generate_test_data():
    """テスト用の入力データをAIで生成する"""
    client = get_client()
    n = random.randint(1, 99999)

    # 業種キーワードと事業形態をランダムに組み合わせて強制的にバリエーションを出す
    products = [
        "寿司・日本料理", "自動車部品", "半導体製造装置", "クラウド会計ソフト", "建設用重機",
        "医療用内視鏡", "冷凍食品", "化粧品", "農業用ドローン", "ホテル運営",
        "語学教育サービス", "物流管理システム", "再生可能エネルギー設備", "ペット用品",
        "ゲームソフト", "工業用接着剤", "歯科医療機器", "ファッションブランド", "保険商品",
        "3Dプリンター", "水処理装置", "フィットネスジム", "印刷機械", "有機食品",
        "カーナビゲーション", "ベアリング", "映像制作サービス", "セキュリティシステム",
        "レストランチェーン", "住宅建材", "電子部品", "コンサルティング", "介護サービス",
        "清掃ロボット", "光ファイバー", "アニメ配信", "産業用ポンプ", "旅行代理店",
        "学習塾", "ワインの輸入販売", "美容機器", "倉庫自動化システム", "翻訳サービス",
    ]
    seed_product = random.choice(products)

    prompt = f"""あなたはテストデータ生成器です。日本企業から米国に赴任するビジネスパーソンの架空プロフィールを1つ生成してください。

この人物の会社の米国子会社が扱う製品・サービスは「{seed_product}」に関連するものにしてください。
乱数: {n}

JSONのみ出力。説明や前置きは一切不要。

{{"bizProduct":"米国子会社が扱う製品・サービス","bizActivity":"事業活動（製造、販売、開発など）","bizCustomer":"顧客・用途（何に使われるか、誰に提供するか）","bizStrength":"米国市場での強み・特徴","changeWhat":"最近の事業環境の変化","changeImpact":"その変化による具体的な影響","challengeGap":"米国側で足りていないこと","challengeWhere":"どの部門でどんな課題を解決する必要があるか","personnelSkill":"必要なスキル・経験","personnelWhy":"現地採用ではなく日本から派遣する理由","usDepartment":"米国赴任先の部門","usPosition":"米国でのポジション","usOfficeRole":"ミッション（「〜する」で終わる目標文。数字は入れない）","domain":"担当する製品・設備・システムの種類（型番は入れない）","department":"日本での部門","position":"日本でのポジション","yearsExperience":"3年未満/3〜10年/10〜20年/20年以上のいずれか","relevantExperience":"ミッション遂行に役立つ過去の経験を2〜3個、読点区切り","job1_period":"1つ目の職歴の期間（例：2015年4月〜2019年3月）","job1_dept":"1つ目の部署名","job1_title":"1つ目の役職","job1_role":"1つ目の主な役割","job1_work":"1つ目の具体的な取り組み","job1_result":"1つ目の成果・結果","job2_period":"2つ目の職歴の期間（例：2019年4月〜現在）","job2_dept":"2つ目の部署名","job2_title":"2つ目の役職","job2_role":"2つ目の主な役割","job2_work":"2つ目の具体的な取り組み","job2_result":"2つ目の成果・結果"}}

ルール:
- 全項目が整合性のあるストーリーになること
- usOfficeRoleに数字・%・期間は入れないこと
- domainに型番やブランド名は入れないこと
- personnelWhyは「日本本社の〜ノウハウを移管するため」のような理由にすること
- 職歴は2つ生成し、job1が新しい方（現在の部署）、job2が古い方にすること
- 各職歴のresultには具体的な数値成果を含めること"""

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            print(f"[test-data attempt {attempt+1}] Response length: {len(text)}")
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                data = json.loads(match.group())
                if "usOfficeRole" in data:
                    data["usOfficeRole"] = clean_mission(data["usOfficeRole"])
                print(f"[test-data] Success: {list(data.keys())}")
                return data
            else:
                print(f"[test-data] No JSON found in: {text[:200]}")
        except anthropic.APIError as e:
            print(f"[test-data] API error: {e}")
            import time
            time.sleep(2)
            continue
        except json.JSONDecodeError as e:
            print(f"[test-data] JSON parse error: {e}")
            print(f"[test-data] Raw text: {text[:300]}")
            import time
            time.sleep(2)
            continue
    print("[test-data] All attempts failed")
    return {"error": "生成に失敗しました"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    client = get_client()
    system_prompt = PROMPTS.get(request.mode, DUTY_SYSTEM_PROMPT)

    def event_stream():
        import time
        for attempt in range(3):
            try:
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=request.messages,
                ) as stream:
                    for text in stream.text_stream:
                        yield f"data: {json.dumps(text)}\n\n"
                yield "data: [DONE]\n\n"
                return
            except anthropic.APIError as e:
                if "overloaded" in str(e).lower() and attempt < 2:
                    time.sleep(3)
                    continue
                yield f"data: [ERROR] {str(e)}\n\n"
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("\n  サポートレター作成支援ツール v41")
    print("  http://localhost:8000 をブラウザで開いてください\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
