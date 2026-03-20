"""
Ultron v3 — Agent-based music production chat

Architecture:
  Phase 1: Brief → Perplexity research + Claude verification (auto)
  Phase 2: Free chat → Intent detection (Haiku) → Agent routing

Agents & Models:
  - Intent detection:  Claude Haiku
  - Title generation:  Claude Haiku
  - Research:          Perplexity (sonar-deep-research → sonar → Claude fallback)
  - Verification:      Claude Sonnet
  - Direction agent:   Claude Sonnet (extended thinking)
  - Lyrics agent:      GPT-4o
  - Suno agent:        GPT-4o
  - General agent:     Claude Sonnet
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

app = FastAPI(title="Ultron v3 — Agent Chat")


@app.on_event("startup")
async def startup():
    db.init_db()


# ========================================================================
# API Clients
# ========================================================================

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
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    finally:
        os.unlink(tmp_path)


def generate_title(brief: str) -> str:
    """Claude Haiku — タイトル生成"""
    cleaned = re.sub(r'https?://\S+', '', brief)
    cleaned = re.sub(r'[_*#\-=]{2,}', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()[:300]
    try:
        response = claude_client.messages.create(
            model="claude-haiku-4-20250414",
            max_tokens=30,
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
        title = response.content[0].text.strip().strip("「」『』\"")
        return title[:15] if title else brief[:40].replace("\n", " ")
    except Exception:
        return brief[:40].replace("\n", " ")


def _parse_json_from_text(raw: str) -> dict | list | None:
    """Extract JSON from text that may contain markdown fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*([\[{].*?[\]}])\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'[\[{].*[\]}]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ========================================================================
# Phase 1: Research + Verification
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

【3. そのユーザー層に響くものは何か？】
- 音楽的特徴: BPM帯、サウンド、曲構成、プロダクションの傾向
- 歌詞テーマ: どんな言葉・メッセージ・ストーリーが刺さるか
- SNSでバイラルした曲の共通要素
- 類似アーティストのヒット曲で参考になるもの

【4. このアーティストの課題は何か？】
- 弱み: ファンや批評家から指摘されている点
- 伸びしろ: まだ攻められていない領域・層
- 競合との差: 同ジャンルの他アーティストとの比較
- 次の一手として狙えるポジション

ルール:
- 具体的な曲名・数字を挙げて根拠を示す
- 日本語で回答
"""


def deep_research(brief: str) -> str:
    """Perplexity research: sonar-deep-research → sonar → Claude fallback"""
    for model in ("sonar-deep-research", "sonar"):
        try:
            response = perplexity_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": f"以下の企画書からアーティスト名を読み取り、4項目を調査してください:\n\n{brief}"},
                ],
            )
            result = response.choices[0].message.content
            prefix = f"[model: {model}]\n\n" if model != "sonar-deep-research" else ""
            return prefix + result
        except Exception:
            continue

    # Claude fallback
    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": (
                    f"{RESEARCH_SYSTEM_PROMPT}\n\n"
                    "※注意: Web検索は使えません。知識ベースに基づいて分析してください。\n\n"
                    f"以下の企画書から調査してください:\n\n{brief}"
                ),
            }],
        )
        return "[model: claude-sonnet (Perplexity unavailable)]\n\n" + response.content[0].text
    except Exception as e:
        raise RuntimeError(f"全リサーチモデルが失敗: {str(e)}")


VERIFICATION_PROMPT = """\
あなたは音楽戦略コンサルタントです。
Perplexityによるリサーチ結果を検証し、プロデューサー向けの戦略ブリーフィングを行ってください。

以下を含めてください:
1. リサーチの信頼性評価（データの裏付けがある点、推測に過ぎない点）
2. 戦略的に最も重要な発見3つ
3. リサーチで見落とされている可能性がある視点
4. 今すぐ使えるアクションヒント

文体: 簡潔、戦略的、データドリブン。箇条書きベースで。
"""


def verify_research(brief: str, research: str) -> str:
    """Claude Sonnet — リサーチ結果を検証・分析"""
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": (
                f"{VERIFICATION_PROMPT}\n\n"
                f"【企画書】\n{brief}\n\n"
                f"【Perplexityリサーチ結果】\n{research}"
            ),
        }],
    )
    return response.content[0].text


SUMMARY_PROMPT = """\
リサーチ結果を以下の3項目に要約してください。
各項目は箇条書き3〜5個でシンプルにまとめる。長い説明は不要。

出力フォーマット（JSON）:
{
  "audience": ["響いているユーザー層のポイント1", "ポイント2", ...],
  "resonance": ["そのユーザーに響くもののポイント1", "ポイント2", ...],
  "challenges": ["課題・伸びしろのポイント1", "ポイント2", ...]
}

JSON のみ出力。日本語で書く。
"""


def generate_summary(research: str) -> dict:
    """GPT-4o-mini — リサーチをJSON要約"""
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
# Intent Detection — Claude Haiku
# ========================================================================

INTENT_KEYWORDS = {
    "direction": ["方向性", "方向", "コンセプト", "候補", "提案して", "アイデア", "どんな曲"],
    "lyrics": ["歌詞", "リリック", "lyrics", "書いて", "作詞"],
    "suno": ["suno", "sunoプロンプト", "プロンプト作", "音楽生成"],
}


def detect_intent(message: str, recent_messages: list[dict] | None = None) -> str:
    """Claude Haiku — ユーザーの意図を判定"""
    msg_lower = message.lower()

    # Fast path: keyword matching
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in msg_lower for k in keywords):
            return intent

    # Slow path: Haiku classification
    try:
        context = ""
        if recent_messages:
            last_3 = recent_messages[-3:]
            context = "\n".join(f"{m['role']}: {m['content'][:100]}" for m in last_3)
            context = f"\n\n最近の会話:\n{context}"

        response = claude_client.messages.create(
            model="claude-haiku-4-20250414",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": (
                    "ユーザーのメッセージの意図を1単語で分類してください:\n"
                    "- direction: 曲の方向性・コンセプト・ジャンルについて\n"
                    "- lyrics: 歌詞の作成・修正\n"
                    "- suno: Suno音楽生成プロンプトの作成\n"
                    "- general: その他\n\n"
                    f"メッセージ: {message}{context}\n\n"
                    "出力（1単語のみ）:"
                ),
            }],
        )
        result = response.content[0].text.strip().lower()
        if result in ("direction", "lyrics", "suno", "general"):
            return result
    except Exception:
        pass
    return "general"


# ========================================================================
# Agents
# ========================================================================

def _build_context(project: dict, messages: list[dict], artifacts: dict) -> str:
    """全エージェント共通のコンテキストを構築"""
    parts = [f"【企画書】\n{project.get('brief', '')}"]
    if project.get("research"):
        parts.append(f"【リサーチ結果】\n{project['research'][:3000]}")
    if artifacts.get("directions"):
        dirs_text = json.dumps(artifacts["directions"], ensure_ascii=False, indent=1)
        parts.append(f"【現在の方向性候補】\n{dirs_text}")
    if artifacts.get("lyrics"):
        lyrics_data = artifacts["lyrics"]
        lyrics_text = lyrics_data.get("lyrics", "") if isinstance(lyrics_data, dict) else str(lyrics_data)
        parts.append(f"【現在の歌詞】\n{lyrics_text[:2000]}")
    if artifacts.get("suno"):
        suno_data = artifacts["suno"]
        suno_text = suno_data.get("prompt", "") if isinstance(suno_data, dict) else str(suno_data)
        parts.append(f"【現在のSunoプロンプト】\n{suno_text}")
    # Recent chat (last 10)
    if messages:
        recent = messages[-10:]
        chat_text = "\n".join(f"{'ユーザー' if m['role']=='user' else 'アシスタント'}: {m['content'][:200]}" for m in recent)
        parts.append(f"【最近の会話】\n{chat_text}")
    return "\n\n".join(parts)


# --- Direction Agent: Claude Sonnet + Extended Thinking ---

DIRECTION_PROMPT = """\
あなたは音楽プロデューサーの戦略パートナーです。
企画書とリサーチ結果を深く分析し、曲の方向性候補を3〜5個提案してください。

各候補は明確に異なるアプローチにする（安全策/挑戦的/トレンド乗り/意外性など）。
リサーチの具体的なデータや事例を根拠に使う。

出力フォーマット — JSON配列のみ:
```json
[
  {
    "title": "方向性の名前（10文字以内）",
    "concept": "コンセプト説明（3-4文。根拠を含む）",
    "reference": "参考曲2-3曲",
    "mood": "ムードキーワード（3-5個、カンマ区切り）",
    "hook": "フック・差別化ポイント（1-2文）",
    "risk": "リスク・注意点（1-2文）",
    "bpm_range": "想定BPM帯",
    "vocal_direction": "ボーカルの方向性"
  }
]
```
JSON配列のみ出力。日本語で書く。
"""

DIRECTION_REFINE_PROMPT = """\
あなたは音楽プロデューサーの戦略パートナーです。
前回の方向性候補に対してフィードバックが来ました。
フィードバックを踏まえて方向性候補を改善・再提案してください。

ルール:
- 必ず3〜5個の候補を出す
- フィードバックの指摘を的確に反映する
- 大きく変えた候補のtitleには末尾に「★」を付ける
- JSON配列のみ出力。日本語で書く。

同じJSON形式で出力してください。
"""


def run_direction_agent(project: dict, messages: list[dict], artifacts: dict, user_message: str) -> dict:
    """Claude Sonnet + Extended Thinking → 方向性候補"""
    has_existing = bool(artifacts.get("directions"))
    prompt = DIRECTION_REFINE_PROMPT if has_existing else DIRECTION_PROMPT
    context = _build_context(project, messages, artifacts)

    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\n{context}\n\n【ユーザーのリクエスト】\n{user_message}",
        }],
    )

    thinking_text = ""
    output_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            output_text = block.text

    parsed = _parse_json_from_text(output_text)
    if isinstance(parsed, list):
        directions = parsed
    elif isinstance(parsed, dict):
        for key in ("directions", "candidates", "items"):
            if key in parsed and isinstance(parsed[key], list):
                directions = parsed[key]
                break
        else:
            directions = [parsed]
    else:
        directions = [{"title": "解析エラー", "concept": output_text[:500], "reference": "", "mood": "", "hook": "", "risk": ""}]

    n = len(directions)
    action = "更新" if has_existing else "提案"
    content = f"🎯 方向性を{n}案{action}しました。右パネルで確認してください。"

    return {
        "content": content,
        "agent": "direction",
        "artifact_type": "directions",
        "artifact": directions,
        "metadata": {"thinking": thinking_text, "model": "claude-sonnet-4 (extended thinking)"},
    }


# --- Lyrics Agent: GPT-4o ---

LYRICS_PROMPT = """\
あなたはプロの作詞家です。
企画書、リサーチ、方向性のコンテキストを踏まえて、曲の歌詞を書いてください。

出力フォーマット（JSON）:
{
    "title": "曲のタイトル",
    "lyrics": "歌詞全文（[Verse 1], [Chorus] 等のセクションラベル付き）",
    "structure": "曲の構成（例: Verse1 / Chorus / Verse2 / Chorus / Bridge / Chorus）",
    "notes": "作詞ノート（狙い、韻の工夫、感情の起伏の説明）"
}

ルール:
- セクションラベルを必ず付ける（[Verse 1], [Pre-Chorus], [Chorus], [Bridge] 等）
- 感情の起伏を意識した構成にする
- 韻を踏む箇所を意識する
- ターゲットリスナーに刺さる言葉を選ぶ
- JSON のみ出力
- 日本語で書く（英語パートがある場合はその旨記載）
"""


def run_lyrics_agent(project: dict, messages: list[dict], artifacts: dict, user_message: str) -> dict:
    """GPT-4o → 歌詞生成"""
    context = _build_context(project, messages, artifacts)
    has_existing = bool(artifacts.get("lyrics"))

    system = LYRICS_PROMPT
    if has_existing:
        system += "\n\n前回の歌詞に対するフィードバックが来ています。フィードバックを反映して歌詞を修正してください。"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n【ユーザーのリクエスト】\n{user_message}"},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = _parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        parsed = {"title": "", "lyrics": raw, "structure": "", "notes": ""}

    action = "修正" if has_existing else "生成"
    title_part = f"「{parsed.get('title', '')}」" if parsed.get("title") else ""
    content = f"✍️ 歌詞を{action}しました{title_part}。右パネルで確認してください。"
    if parsed.get("notes"):
        content += f"\n\n📝 {parsed['notes']}"

    return {
        "content": content,
        "agent": "lyrics",
        "artifact_type": "lyrics",
        "artifact": parsed,
        "metadata": {"model": "gpt-4o"},
    }


# --- Suno Agent: GPT-4o ---

SUNO_PROMPT = """\
あなたはSunoの音楽生成プロンプトの専門家です。
企画のコンテキストを踏まえて、最適なSunoプロンプトを作成してください。

出力フォーマット（JSON）:
{
    "style": "Style of Music（英語で。ジャンル、ムード、楽器を含む詳細な説明）",
    "title": "曲のタイトル",
    "tags": ["tag1", "tag2", ...],
    "lyrics": "Suno用フォーマットの歌詞（[Verse], [Chorus], [Bridge], [Outro] 等のタグ付き）",
    "negative_tags": ["避けるスタイル1", ...],
    "notes": "プロンプト設計ノート（なぜこのスタイルを選んだか）"
}

ルール:
- style は英語で、具体的かつ詳細に記述（30〜60語）
- tags は英語で5〜10個
- 歌詞がある場合はSunoフォーマット（[Verse], [Chorus]等）に変換
- 歌詞がない場合は [Instrumental] と記載
- negative_tags で避けるべきスタイルを明記
- JSON のみ出力
"""


def run_suno_agent(project: dict, messages: list[dict], artifacts: dict, user_message: str) -> dict:
    """GPT-4o → Sunoプロンプト生成"""
    context = _build_context(project, messages, artifacts)
    has_existing = bool(artifacts.get("suno"))

    system = SUNO_PROMPT
    if has_existing:
        system += "\n\n前回のSunoプロンプトに対するフィードバックが来ています。修正してください。"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n【ユーザーのリクエスト】\n{user_message}"},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = _parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        parsed = {"style": raw, "title": "", "tags": [], "lyrics": "", "negative_tags": [], "notes": ""}

    action = "更新" if has_existing else "生成"
    content = f"🎵 Sunoプロンプトを{action}しました。右パネルで確認してください。"
    if parsed.get("notes"):
        content += f"\n\n💡 {parsed['notes']}"

    return {
        "content": content,
        "agent": "suno",
        "artifact_type": "suno",
        "artifact": parsed,
        "metadata": {"model": "gpt-4o"},
    }


# --- General Agent: Claude Sonnet ---

def run_general_agent(project: dict, messages: list[dict], artifacts: dict, user_message: str) -> dict:
    """Claude Sonnet — 一般的な質問に回答"""
    context = _build_context(project, messages, artifacts)

    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=(
            "あなたは音楽プロデューサーの戦略パートナーです。"
            "プロジェクトのコンテキストを踏まえて、簡潔かつ実用的に回答してください。"
            "必要に応じて方向性・歌詞・Sunoプロンプトの生成を提案してください。"
        ),
        messages=[{
            "role": "user",
            "content": f"{context}\n\n【質問】\n{user_message}",
        }],
    )

    return {
        "content": response.content[0].text,
        "agent": "general",
        "artifact_type": None,
        "artifact": None,
        "metadata": {"model": "claude-sonnet-4"},
    }


# ========================================================================
# API Endpoints
# ========================================================================

@app.post("/api/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """Phase 1: 企画書 → リサーチ + 検証 + サマリー"""
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
        # 1. Title (Haiku)
        title = generate_title(brief)

        # 2. Research (Perplexity)
        research = deep_research(brief)

        # 3. Verification (Claude Sonnet)
        verification = verify_research(brief, research)

        # 4. Summary (GPT-4o-mini)
        summary = generate_summary(research)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"分析中にエラー: {str(e)}"})

    # Save project
    now = datetime.now().isoformat()
    project_id = str(uuid.uuid4())
    entry = {
        "id": project_id,
        "title": title,
        "created_at": now,
        "brief": brief,
        "research": research,
    }
    db.save_project(entry)

    # Save initial messages
    db.save_message(project_id, "user", brief[:500] + ("..." if len(brief) > 500 else ""))
    analysis_msg = db.save_message(
        project_id, "assistant", verification,
        agent="research",
        metadata={"summary": summary, "model": "perplexity + claude-sonnet"},
    )

    return {
        "project": {"id": project_id, "title": title, "created_at": now, "brief": brief},
        "research": research,
        "verification": verification,
        "summary": summary,
        "messages": [
            {"role": "user", "content": brief[:500] + ("..." if len(brief) > 500 else "")},
            {
                "role": "assistant", "content": verification,
                "agent": "research",
                "metadata": {"summary": summary, "model": "perplexity + claude-sonnet"},
            },
        ],
    }


class ChatRequest(BaseModel):
    project_id: str
    message: str


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Phase 2: ユーザーメッセージ → Intent検出 → Agent実行"""
    project = db.get_project(req.project_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "プロジェクトが見つかりません"})

    # Save user message
    db.save_message(req.project_id, "user", req.message)

    # Load context
    messages = db.get_messages(req.project_id)
    artifacts = db.get_latest_artifacts(req.project_id)

    # Detect intent
    intent = detect_intent(req.message, messages)

    # Route to agent
    try:
        if intent == "direction":
            result = run_direction_agent(project, messages, artifacts, req.message)
        elif intent == "lyrics":
            result = run_lyrics_agent(project, messages, artifacts, req.message)
        elif intent == "suno":
            result = run_suno_agent(project, messages, artifacts, req.message)
        else:
            result = run_general_agent(project, messages, artifacts, req.message)
    except Exception as e:
        error_msg = f"エージェントエラー ({intent}): {str(e)}"
        db.save_message(req.project_id, "assistant", error_msg, agent=intent)
        return JSONResponse(status_code=500, content={"error": error_msg})

    # Save assistant message
    db.save_message(
        req.project_id, "assistant", result["content"],
        agent=result["agent"],
        artifact_type=result.get("artifact_type"),
        artifact=result.get("artifact"),
        metadata=result.get("metadata"),
    )

    return {
        "message": {
            "role": "assistant",
            "content": result["content"],
            "agent": result["agent"],
            "artifact_type": result.get("artifact_type"),
            "artifact": result.get("artifact"),
            "metadata": result.get("metadata", {}),
        },
    }


# ========================================================================
# History
# ========================================================================

@app.get("/api/history")
async def get_history():
    return db.load_history()


@app.get("/api/history/{entry_id}")
async def get_history_entry(entry_id: str):
    project = db.get_project(entry_id)
    if not project:
        return JSONResponse(status_code=404, content={"error": "見つかりません"})
    messages = db.get_messages(entry_id)
    artifacts = db.get_latest_artifacts(entry_id)
    return {
        "project": project,
        "messages": messages,
        "artifacts": artifacts,
    }


@app.delete("/api/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    db.delete_project(entry_id)
    return {"ok": True}


# ========================================================================
# Static files
# ========================================================================

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5015)
