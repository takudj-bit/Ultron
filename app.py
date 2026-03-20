"""
Ultron Web App - 音楽プロデューサー向け歌詞制作ワークフロー (Web版)
FastAPI backend: 企画書アップロード/テキスト入力 → Perplexityリサーチ → GPT-4o歌詞ドラフト
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env", override=True)

app = FastAPI(title="Ultron - 歌詞制作ワークフロー")

HISTORY_FILE = Path(__file__).parent / "data" / "history.json"
HISTORY_FILE.parent.mkdir(exist_ok=True)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text("utf-8"))
    return []


def save_history(history: list[dict]):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
perplexity_client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _clean_brief_for_title(brief: str) -> str:
    """タイトル生成用にURLや記号を除去して先頭部分だけ取得"""
    import re
    text = re.sub(r'https?://\S+', '', brief)  # URL除去
    text = re.sub(r'[_*#\-=]{2,}', '', text)   # 装飾記号除去
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300]


def generate_title(brief: str) -> str:
    """GPT-4o-miniで履歴タイトルを自動生成（安くて速い）"""
    cleaned = _clean_brief_for_title(brief)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "音楽企画書の内容から、曲のプロジェクト名を考えて。\n"
                        "ルール:\n"
                        "- 日本語10文字以内\n"
                        "- タイトルのみ出力（説明不要）\n"
                        "- 「」や記号は付けない\n\n"
                        f"{cleaned}"
                    ),
                }
            ],
        )
        title = response.choices[0].message.content.strip().strip("「」『』\"")
        return title[:15] if title else brief[:40].replace("\n", " ")
    except Exception:
        return brief[:40].replace("\n", " ")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """PDFバイトからテキストを抽出"""
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


def research_trends(brief: str) -> str:
    """Perplexity APIでトレンドリサーチ"""
    response = perplexity_client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは音楽業界のトレンドリサーチャーです。"
                    "企画書の内容を分析し、以下を調査してください:\n"
                    "1. 該当ジャンルの最新トレンド（テーマ、言葉遣い、表現手法）\n"
                    "2. 類似コンセプトのヒット曲とその歌詞の特徴\n"
                    "3. ターゲット層に響くキーワードやフレーズ\n"
                    "4. 避けるべき表現やクリシェ\n"
                    "日本語で回答してください。"
                ),
            },
            {
                "role": "user",
                "content": f"以下の企画書に基づいてトレンドリサーチしてください:\n\n{brief}",
            },
        ],
    )
    return response.choices[0].message.content


def distill_research(brief: str, research: str) -> str:
    """リサーチ結果を歌詞制作に関連するポイントだけに要約"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは音楽プロデューサーのアシスタントです。\n"
                    "トレンドリサーチの結果を受け取り、歌詞制作に直接役立つ情報だけを抽出・要約してください。\n\n"
                    "出力フォーマット:\n"
                    "【使えるキーワード・フレーズ】箇条書き\n"
                    "【参考にすべき表現手法】箇条書き\n"
                    "【避けるべき表現】箇条書き\n"
                    "【トーン・ムードの方向性】1-2文\n\n"
                    "ルール:\n"
                    "- 業界分析や市場データは省く\n"
                    "- 歌詞に使える具体的な言葉・比喩・表現に焦点を当てる\n"
                    "- 企画書のコンセプトとの整合性を意識する\n"
                    "- 簡潔に。各セクション最大5項目まで\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【企画書】\n{brief}\n\n"
                    f"【トレンドリサーチ結果】\n{research}\n\n"
                    "上記から歌詞制作に必要なポイントだけを抽出してください。"
                ),
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def draft_lyrics(brief: str, research: str) -> str:
    """GPT-4oで歌詞ドラフトを生成"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたはプロの作詞家です。\n"
                    "企画書の意図とトレンドリサーチの結果を踏まえ、歌詞ドラフトを作成してください。\n\n"
                    "ルール:\n"
                    "- 企画書のコンセプト・世界観を忠実に反映する\n"
                    "- トレンドを取り入れつつオリジナリティを出す\n"
                    "- Aメロ / Bメロ / サビ / (Cメロ) の構成で書く\n"
                    "- 各セクションにメロディの方向性メモを添える\n"
                    "- 歌詞の後に、意図・狙いの解説を付ける\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【企画書】\n{brief}\n\n"
                    f"【トレンドリサーチ結果】\n{research}\n\n"
                    "これらを踏まえて歌詞ドラフトを書いてください。"
                ),
            },
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content


@app.post("/api/generate")
async def generate(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """企画書からトレンドリサーチ + 歌詞ドラフトを生成"""
    # 入力テキストを取得
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
        return JSONResponse(
            status_code=400,
            content={"error": "ファイルまたはテキストを入力してください"},
        )

    try:
        research = research_trends(brief)
        distilled = distill_research(brief, research)
        lyrics = draft_lyrics(brief, distilled)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"生成中にエラーが発生しました: {str(e)}"},
        )

    # 履歴に保存
    now = datetime.now().isoformat()
    title = generate_title(brief)
    entry = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": now,
        "brief": brief,
        "research": research,
        "distilled": distilled,
        "lyrics": lyrics,
        "versions": [
            {"label": "v1", "lyrics": lyrics, "parent": None, "feedback": None, "timestamp": now}
        ],
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history)

    return {
        "id": entry["id"],
        "brief": brief[:500] + ("..." if len(brief) > 500 else ""),
        "research": research,
        "distilled": distilled,
        "lyrics": lyrics,
    }


@app.get("/api/history")
async def get_history():
    return load_history()


@app.post("/api/history/regenerate-titles")
async def regenerate_titles():
    """既存プロジェクトのタイトルをClaude Haikuで再生成"""
    history = load_history()
    updated = 0
    for entry in history:
        brief = entry.get("brief", "")
        if brief:
            new_title = generate_title(brief)
            if new_title and new_title != entry.get("title"):
                entry["title"] = new_title
                updated += 1
    save_history(history)
    return {"updated": updated, "total": len(history)}


@app.get("/api/history/{entry_id}")
async def get_history_entry(entry_id: str):
    for entry in load_history():
        if entry["id"] == entry_id:
            return entry
    return JSONResponse(status_code=404, content={"error": "見つかりません"})


@app.delete("/api/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    history = load_history()
    history = [e for e in history if e["id"] != entry_id]
    save_history(history)
    return {"ok": True}


class ReviseRequest(BaseModel):
    entry_id: str
    current_lyrics: str
    feedback: str
    from_version: str = ""  # e.g. "v1", "v2", "v3-a"


@app.post("/api/revise")
async def revise_lyrics(req: ReviseRequest):
    """フィードバックに基づいて歌詞を修正"""
    # 履歴からコンテキストを取得
    brief = ""
    distilled = ""
    for entry in load_history():
        if entry["id"] == req.entry_id:
            brief = entry.get("brief", "")
            distilled = entry.get("distilled", entry.get("research", ""))
            break

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたはプロの作詞家です。\n"
                        "ユーザーが歌詞のドラフトに対するフィードバックを送ります。\n"
                        "フィードバックに基づいて歌詞を修正してください。\n\n"
                        "ルール:\n"
                        "- フィードバックで指摘された箇所を的確に修正する\n"
                        "- 指摘されていない良い部分はなるべく維持する\n"
                        "- 修正後の歌詞全文を出力する\n"
                        "- 最後に修正箇所の簡潔な説明を付ける\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"【企画書】\n{brief}\n\n"
                        f"【リサーチ要約】\n{distilled}\n\n"
                        f"【現在の歌詞】\n{req.current_lyrics}\n\n"
                        f"【修正リクエスト】\n{req.feedback}\n\n"
                        "上記のフィードバックに基づいて歌詞を修正してください。"
                    ),
                },
            ],
            temperature=0.7,
        )
        revised = response.choices[0].message.content
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"修正中にエラー: {str(e)}"},
        )

    # バージョンラベルを計算
    history = load_history()
    for entry in history:
        if entry["id"] == req.entry_id:
            if "versions" not in entry:
                # 既存データをマイグレーション
                entry["versions"] = [{"label": "v1", "lyrics": entry.get("lyrics", ""), "parent": None, "feedback": None, "timestamp": entry.get("created_at", "")}]
                # 旧revisionsがあればマイグレーション
                for i, rev in enumerate(entry.get("revisions", [])):
                    entry["versions"].append({"label": f"v{i+2}", "lyrics": rev.get("previous_lyrics", ""), "parent": f"v{i+1}", "feedback": rev.get("feedback", ""), "timestamp": rev.get("timestamp", "")})

            versions = entry["versions"]
            from_ver = req.from_version or versions[-1]["label"]

            # 次のラベルを決定
            # from_verの直接の子がすでにあるか?
            children = [v for v in versions if v.get("parent") == from_ver]
            if not children:
                # まだ子がない → 連番
                # from_verが"v3"なら"v4", "v3-a"なら"v3-a.1"的に
                base_num = len([v for v in versions if "-" not in v["label"] and v["label"].startswith("v")])
                new_label = f"v{base_num + 1}"
            else:
                # すでに子がある → 分岐: v3-a, v3-b...
                branch_children = [v for v in versions if v.get("parent") == from_ver and "-" in v["label"].split("v")[-1]]
                if branch_children:
                    # 既に分岐がある → 次のアルファベット
                    last_suffix = sorted([v["label"] for v in branch_children])[-1]
                    last_char = last_suffix.split("-")[-1]
                    next_char = chr(ord(last_char) + 1)
                    new_label = f"{from_ver}-{next_char}"
                else:
                    # 最初の分岐
                    new_label = f"{from_ver}-a"

            versions.append({
                "label": new_label,
                "lyrics": revised,
                "parent": from_ver,
                "feedback": req.feedback,
                "timestamp": datetime.now().isoformat(),
            })
            entry["lyrics"] = revised
            entry["versions"] = versions

            # 旧revisions互換
            if "revisions" not in entry:
                entry["revisions"] = []
            entry["revisions"].append({
                "feedback": req.feedback,
                "previous_lyrics": req.current_lyrics,
                "timestamp": datetime.now().isoformat(),
            })
            break
    save_history(history)

    return {"lyrics": revised, "version_label": new_label}


# 静的ファイルの配信（フロントエンド）
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5015)
