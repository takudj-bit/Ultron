"""
Ultron v2 — 音楽プロデューサー向け制作ワークフロー (5-Step)

① クライアント企画書インプット
② Perplexity deep-research → サマリー確認フェーズ
③ Claude extended thinking → 方向性候補を3〜5個提示
④ ユーザーが方向性を選択（未実装）
⑤ Sunoプロンプト＋歌詞候補生成（未実装）
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env", override=True)

import db  # noqa: E402

app = FastAPI(title="Ultron v2 — 制作ワークフロー")


@app.on_event("startup")
async def startup():
    db.init_db()


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
perplexity_client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ========================================================================
# Utility
# ========================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    finally:
        os.unlink(tmp_path)


def _clean_brief_for_title(brief: str) -> str:
    text = re.sub(r'https?://\S+', '', brief)
    text = re.sub(r'[_*#\-=]{2,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300]


def generate_title(brief: str) -> str:
    cleaned = _clean_brief_for_title(brief)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": (
                    "音楽企画書の内容から、曲のプロジェクト名を考えて。\n"
                    "ルール:\n- 日本語10文字以内\n- タイトルのみ出力（説明不要）\n"
                    "- 「」や記号は付けない\n\n"
                    f"{cleaned}"
                ),
            }],
        )
        title = response.choices[0].message.content.strip().strip("「」『』\"")
        return title[:15] if title else brief[:40].replace("\n", " ")
    except Exception:
        return brief[:40].replace("\n", " ")


def _parse_directions_json(raw: str) -> list[dict]:
    """Claude/GPTの出力からJSON配列をパースする"""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("directions", "candidates", "options", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            if all(isinstance(v, dict) for v in parsed.values()):
                return list(parsed.values())
            return [parsed]
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return [{"title": "解析エラー", "concept": raw[:500], "reference": "", "mood": "", "hook": "", "risk": ""}]


# ========================================================================
# Step ② — Perplexity Deep Research
# ========================================================================

RESEARCH_SYSTEM_PROMPT = """\
あなたは音楽業界の戦略リサーチャーです。
企画書からアーティスト名を読み取り、以下の4項目を必ず調査してください。
アーティストが複数いる場合はそれぞれについて調査してください。

【1. このアーティストは誰か？】
- 経歴・ジャンル・所属レーベル
- 代表曲トップ3〜5（再生数・チャート実績を含む）
- 現在の市場ポジション（上昇期/安定期/再起期）
- 他のアーティストにない強み・武器
- 最近のリリースやプロジェクトの評価

【2. このアーティストに響いているユーザー層は誰か？】
- ファン層の年齢・性別・地域の分布
- ファンの感情的な反応傾向（何に共感し、どこで熱狂するか）
- ファンのSNS行動パターン（推し活・二次創作・ミーム・拡散行動）
- リスナーがこのアーティストに求めているもの
- ライブ/イベントでの反応の特徴

【3. そのユーザー層に響くものは何か？】
- 音楽的特徴: BPM帯、サウンド、曲構成、プロダクションの傾向
- 歌詞テーマ: どんな言葉・メッセージ・ストーリーが刺さるか
- ビジュアル/MV傾向: 世界観、色味、演出で反応が良いもの
- SNSでバイラルした曲の共通要素
- 類似アーティストのヒット曲で参考になるもの（曲名・理由を添えて）

【4. このアーティストの課題は何か？】
- 弱み: ファンや批評家から指摘されている点
- 伸びしろ: まだ攻められていない領域・ジャンル・層
- 競合との差: 同ジャンルの他アーティストとの比較で負けている点
- リスク: 避けるべき方向性・飽和している領域
- 次の一手として狙えるポジション

ルール:
- 企画書内のアーティスト名を正確に読み取って調査する
- 具体的な曲名・アーティスト名・数字を挙げて根拠を示す
- 数字（再生数、チャート順位、フォロワー数等）を可能な限り含める
- 日本語で回答
- 各セクションを見出しで明確に分ける
- 推測と事実を区別する（「推定」「不明」と明記）
"""


def deep_research(brief: str) -> str:
    """Perplexity research: deep-research → sonar → Claude フォールバック"""
    for model in ("sonar-deep-research", "sonar"):
        try:
            response = perplexity_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "以下の企画書からアーティスト名を読み取り、4項目を調査してください:\n\n"
                            f"{brief}"
                        ),
                    },
                ],
            )
            result = response.choices[0].message.content
            prefix = f"[model: {model}]\n\n" if model != "sonar-deep-research" else ""
            return prefix + result
        except Exception:
            continue

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": (
                    f"{RESEARCH_SYSTEM_PROMPT}\n\n"
                    "※注意: Web検索は使えません。知識ベースに基づいて分析してください。\n\n"
                    "以下の企画書からアーティスト名を読み取り、4項目を調査してください:\n\n"
                    f"{brief}"
                ),
            }],
        )
        return "[model: claude-sonnet (Perplexity quota exceeded)]\n\n" + response.content[0].text
    except Exception as e:
        raise RuntimeError(f"全リサーチモデルが失敗: {str(e)}")


# ========================================================================
# Step ②' — リサーチ結果のサマリー生成
# ========================================================================

SUMMARY_PROMPT = """\
リサーチ結果を以下の3項目に要約してください。
各項目は箇条書き3〜5個でシンプルにまとめる。長い説明は不要。

出力フォーマット（JSON）:
{
  "audience": ["響いているユーザー層のポイント1", "ポイント2", ...],
  "resonance": ["そのユーザーに響くもののポイント1", "ポイント2", ...],
  "challenges": ["課題・伸びしろのポイント1", "ポイント2", ...]
}

ルール:
- 各項目は最大5個まで
- 1つのポイントは1〜2文
- JSON のみ出力。説明文は不要
- 日本語で書く
"""


def generate_summary(research: str) -> dict:
    """リサーチ結果を3項目サマリーに要約"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"以下のリサーチ結果を要約してください:\n\n{research}"},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        return {
            "audience": parsed.get("audience", []),
            "resonance": parsed.get("resonance", []),
            "challenges": parsed.get("challenges", []),
        }
    except json.JSONDecodeError:
        return {"audience": ["要約生成エラー"], "resonance": [], "challenges": []}


# ========================================================================
# Step ③ — Claude Extended Thinking → 方向性候補の生成
# ========================================================================

DIRECTIONS_PROMPT = """\
あなたは音楽プロデューサーの戦略パートナーです。

企画書とリサーチ結果を深く分析し、曲の方向性候補を3〜5個提案してください。

思考プロセス:
1. まずリサーチ結果の重要ポイントを整理する
2. アーティストの強みとIPの特性の交差点を見つける
3. ターゲットリスナーの嗜好とトレンドを考慮する
4. 各候補が明確に異なるリスク/リターンのバランスを持つようにする
5. 「なぜこの方向性がハマるか」の根拠をリサーチから導く

最終出力は以下のJSON配列のみ（思考過程は出力しない）:
```json
[
  {
    "title": "方向性の名前（10文字以内）",
    "concept": "コンセプトの説明（3-4文。なぜこの方向性が有効かの根拠を含む）",
    "reference": "参考曲: アーティスト名「曲名」など2-3曲（なぜ参考になるか一言添える）",
    "mood": "ムード・トーンのキーワード（3-5個、カンマ区切り）",
    "hook": "この方向性の核となるフック・差別化ポイント（1-2文）",
    "risk": "リスクや注意点（1-2文）",
    "bpm_range": "想定BPM帯（例: 128-140）",
    "vocal_direction": "ボーカルの方向性（例: エモーショナルなファルセット、ラップとの掛け合いなど）"
  }
]
```

ルール:
- 必ず3〜5個の候補を出す
- 各候補は明確に異なるアプローチにする（安全策/挑戦的/トレンド乗り/意外性/ハイブリッドなど）
- リサーチの具体的なデータや事例を根拠に使う
- JSON配列のみ出力。思考過程やコメントはJSON外に出さない
- 日本語で書く
"""


def generate_directions(brief: str, research: str) -> tuple[list[dict], str]:
    """Claude extended thinkingで方向性候補を生成。(directions, thinking)を返す"""
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{
            "role": "user",
            "content": (
                f"{DIRECTIONS_PROMPT}\n\n"
                f"【企画書】\n{brief}\n\n"
                f"【リサーチ結果】\n{research}\n\n"
                "これらを深く分析した上で、曲の方向性候補をJSON配列で提案してください。"
            ),
        }],
    )
    thinking_text = ""
    output_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            output_text = block.text
    return _parse_directions_json(output_text), thinking_text


# ========================================================================
# Step ③' — 反論を受けてリファイン
# ========================================================================

REFINE_PROMPT = """\
あなたは音楽プロデューサーの戦略パートナーです。
前回提案した方向性候補に対して、プロデューサーからフィードバック（反論・追加要望・修正指示）が来ました。

思考プロセス:
1. フィードバックの本質を読み取る（表面的な言葉の裏にある意図）
2. 現在の候補のどこが良くてどこがダメかを判断する
3. リサーチデータで裏付けながら、新しい候補を考える
4. プロデューサーの感覚を尊重しつつ、データに基づく反論があれば提示する

最終出力は以下のJSON配列のみ:
```json
[
  {
    "title": "方向性の名前（10文字以内、大幅変更なら末尾に★）",
    "concept": "コンセプトの説明（3-4文）",
    "reference": "参考曲: アーティスト名「曲名」など2-3曲",
    "mood": "ムード・トーンのキーワード（3-5個、カンマ区切り）",
    "hook": "この方向性の核となるフック・差別化ポイント（1-2文）",
    "risk": "リスクや注意点（1-2文）",
    "bpm_range": "想定BPM帯",
    "vocal_direction": "ボーカルの方向性"
  }
]
```

ルール:
- 必ず3〜5個の候補を出す
- フィードバックの指摘を的確に反映する
- 良い候補はベースを残しつつ改善、ダメな候補は入れ替える
- 大きく変えた候補のtitleには末尾に「★」を付ける
- JSON配列のみ出力
- 日本語で書く
"""


def refine_directions(brief: str, research: str, current_directions: list[dict], feedback: str) -> tuple[list[dict], str]:
    """Claude extended thinkingで方向性をリファイン"""
    dirs_text = json.dumps(current_directions, ensure_ascii=False, indent=2)
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{
            "role": "user",
            "content": (
                f"{REFINE_PROMPT}\n\n"
                f"【企画書】\n{brief}\n\n"
                f"【リサーチ結果】\n{research}\n\n"
                f"【前回の方向性候補】\n{dirs_text}\n\n"
                f"【プロデューサーのフィードバック】\n{feedback}\n\n"
                "フィードバックを深く分析した上で、方向性候補を改善・再提案してください。JSON配列のみ出力。"
            ),
        }],
    )
    thinking_text = ""
    output_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            output_text = block.text
    return _parse_directions_json(output_text), thinking_text


# ========================================================================
# API Endpoints
# ========================================================================

@app.post("/api/generate")
async def generate(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """Steps ①②: 企画書 → リサーチ → サマリー確認フェーズで止まる"""
    brief = None
    if file and file.filename:
        file_bytes = await file.read()
        if file.filename.lower().endswith(".pdf"):
            brief = extract_text_from_pdf(file_bytes)
        else:
            brief = file_bytes.decode("utf-8")
    elif text and text.strip():
        brief = text.strip()

    if not brief:
        return JSONResponse(status_code=400, content={"error": "ファイルまたはテキストを入力してください"})

    try:
        research = deep_research(brief)
        summary = generate_summary(research)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"リサーチ中にエラー: {str(e)}"})

    now = datetime.now().isoformat()
    title = generate_title(brief)
    entry = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": now,
        "step": 2,
        "brief": brief,
        "research": research,
        "summary": summary,
        "directions": [],
        "selected_direction": None,
        "lyrics": "",
        "suno_prompt": "",
    }
    db.save_project(entry)

    return {
        "id": entry["id"],
        "title": title,
        "brief": brief[:500] + ("..." if len(brief) > 500 else ""),
        "research": research,
        "summary": summary,
        "step": 2,
    }


class ProceedRequest(BaseModel):
    entry_id: str


@app.post("/api/proceed")
async def proceed(req: ProceedRequest):
    """サマリー確認OK → Step③ 方向性候補を生成"""
    entry = db.get_project(req.entry_id)
    if not entry:
        return JSONResponse(status_code=404, content={"error": "プロジェクトが見つかりません"})

    brief = entry.get("brief", "")
    research = entry.get("research", "")

    try:
        directions, thinking = generate_directions(brief, research)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"方向性生成エラー: {str(e)}"})

    db.update_project(req.entry_id, step=3, directions=directions)

    return {
        "directions": directions,
        "thinking": thinking,
        "step": 3,
    }


class ReviseResearchRequest(BaseModel):
    entry_id: str
    feedback: str


@app.post("/api/revise-research")
async def revise_research(req: ReviseResearchRequest):
    """サマリーへの修正コメントを受けてリサーチ補足→サマリー再生成"""
    entry = db.get_project(req.entry_id)
    if not entry:
        return JSONResponse(status_code=404, content={"error": "プロジェクトが見つかりません"})

    research = entry.get("research", "")

    try:
        # Claudeでリサーチを補足・修正
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=6000,
            messages=[{
                "role": "user",
                "content": (
                    "以下はアーティストのリサーチ結果です。\n"
                    "プロデューサーから修正コメントが来ました。\n"
                    "コメントを踏まえてリサーチ結果を補足・修正してください。\n"
                    "元のリサーチの良い部分は残しつつ、指摘された点を改善してください。\n"
                    "修正後のリサーチ全文を出力してください。\n\n"
                    f"【元のリサーチ】\n{research}\n\n"
                    f"【修正コメント】\n{req.feedback}"
                ),
            }],
        )
        revised_research = response.content[0].text
        summary = generate_summary(revised_research)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"リサーチ修正エラー: {str(e)}"})

    db.update_project(req.entry_id, research=revised_research, summary=summary)

    return {
        "research": revised_research,
        "summary": summary,
        "step": 2,
    }


class RefineRequest(BaseModel):
    entry_id: str
    feedback: str


@app.post("/api/refine")
async def refine(req: RefineRequest):
    """方向性候補への反論・フィードバックを受けてリファイン"""
    entry = db.get_project(req.entry_id)
    if not entry:
        return JSONResponse(status_code=404, content={"error": "プロジェクトが見つかりません"})

    try:
        new_dirs, thinking = refine_directions(
            entry.get("brief", ""),
            entry.get("research", ""),
            entry.get("directions", []),
            req.feedback,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"リファイン中にエラー: {str(e)}"})

    db.update_project(req.entry_id, directions=new_dirs)

    return {"directions": new_dirs, "thinking": thinking, "feedback": req.feedback}


# ========================================================================
# History
# ========================================================================

@app.get("/api/history")
async def get_history():
    return db.load_history()


@app.get("/api/history/{entry_id}")
async def get_history_entry(entry_id: str):
    entry = db.get_project(entry_id)
    if entry:
        return entry
    return JSONResponse(status_code=404, content={"error": "見つかりません"})


@app.delete("/api/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    db.delete_project(entry_id)
    return {"ok": True}


@app.post("/api/history/regenerate-titles")
async def regenerate_titles():
    history = db.load_history()
    updated = 0
    for entry in history:
        brief = entry.get("brief", "")
        if brief:
            new_title = generate_title(brief)
            if new_title and new_title != entry.get("title"):
                db.update_project(entry["id"], title=new_title)
                updated += 1
    return {"updated": updated, "total": len(history)}


# ========================================================================
# Static files
# ========================================================================

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5015)
