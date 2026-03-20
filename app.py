"""
Ultron v2 — 音楽プロデューサー向け制作ワークフロー (5-Step)

① クライアント企画書インプット
② Perplexity deep-research（ポジション・響き方・不足点）
③ 曲の方向性候補を3〜5個提示
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

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

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


# ========================================================================
# Step ② — Perplexity Deep Research
# ========================================================================

def deep_research(brief: str) -> str:
    """Perplexity deep-research: アーティストのポジション・響き方・不足点を分析"""
    response = perplexity_client.chat.completions.create(
        model="sonar-deep-research",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは音楽業界の戦略リサーチャーです。\n"
                    "クライアントの企画書を分析し、以下の3軸で徹底的に調査してください。\n\n"
                    "【1. ポジション分析】\n"
                    "- このアーティスト/プロジェクトの現在の市場ポジション\n"
                    "- 競合アーティストとの差別化ポイント\n"
                    "- ジャンル内での立ち位置と強み\n\n"
                    "【2. 響き方分析】\n"
                    "- ターゲットリスナーに何がどう響いているか\n"
                    "- SNS・ストリーミングでの反応パターン\n"
                    "- ファンが求めている要素、感情的フック\n"
                    "- 類似コンセプトの成功曲とその理由\n\n"
                    "【3. 不足点・機会分析】\n"
                    "- 現状で足りていない要素、攻められていない領域\n"
                    "- トレンドとのギャップ\n"
                    "- 次の一手として狙えるポジション\n"
                    "- 避けるべきリスクや飽和領域\n\n"
                    "ルール:\n"
                    "- 具体的な曲名・アーティスト名を挙げて根拠を示す\n"
                    "- 数字やデータがあれば含める\n"
                    "- 日本語で回答\n"
                    "- 各セクションを明確に分けて出力"
                ),
            },
            {
                "role": "user",
                "content": f"以下のクライアント企画書を分析してください:\n\n{brief}",
            },
        ],
    )
    return response.choices[0].message.content


# ========================================================================
# Step ③ — 方向性候補の生成
# ========================================================================

DIRECTIONS_SYSTEM_PROMPT = """\
あなたは音楽プロデューサーの戦略パートナーです。
企画書とリサーチ結果を踏まえ、曲の方向性候補を3〜5個提案してください。

各候補は以下のフォーマットで出力（JSON配列）:
[
  {
    "title": "方向性の名前（10文字以内）",
    "concept": "コンセプトの説明（2-3文）",
    "reference": "参考曲: アーティスト名「曲名」など1-2曲",
    "mood": "ムード・トーンのキーワード（3-5個、カンマ区切り）",
    "hook": "この方向性の核となるフック・差別化ポイント（1文）",
    "risk": "リスクや注意点（1文）"
  }
]

ルール:
- 必ず3〜5個の候補を出す
- 各候補は明確に異なるアプローチにする（安全策、挑戦的、トレンド乗り、意外性など）
- JSON配列のみ出力。説明文は不要
- 日本語で書く
"""


def generate_directions(brief: str, research: str) -> list[dict]:
    """GPT-4oで方向性候補を3〜5個生成"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": DIRECTIONS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【企画書】\n{brief}\n\n"
                    f"【リサーチ結果】\n{research}\n\n"
                    "これらを踏まえて曲の方向性候補を提案してください。"
                ),
            },
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        # Handle both {"directions": [...]} and [...]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("directions", "candidates", "options", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            # If dict has numbered keys or single direction
            return list(parsed.values()) if all(isinstance(v, dict) for v in parsed.values()) else [parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON array from text
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return [{"title": "解析エラー", "concept": raw, "reference": "", "mood": "", "hook": "", "risk": ""}]


# ========================================================================
# API Endpoints — Steps ①②③
# ========================================================================

@app.post("/api/generate")
async def generate(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """Steps ①②③: 企画書 → deep-research → 方向性候補"""
    # ① Brief input
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
        # ② Deep research
        research = deep_research(brief)
        # ③ Direction candidates
        directions = generate_directions(brief, research)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"生成中にエラーが発生しました: {str(e)}"})

    # Save
    now = datetime.now().isoformat()
    title = generate_title(brief)
    entry = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": now,
        "step": 3,
        "brief": brief,
        "research": research,
        "directions": directions,
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
        "directions": directions,
        "step": 3,
    }


# ========================================================================
# API Endpoints — History
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
