"""
Ultron - 音楽プロデューサー向け歌詞制作ワークフロー
企画書(PDF/TXT) or テキスト入力 → Perplexityトレンドリサーチ → GPT-4o歌詞ドラフト
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
perplexity_client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)


def read_input(input_path: str) -> str:
    """PDF/テキストファイルから企画書を読み込み"""
    path = Path(input_path)
    if not path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    else:
        text = path.read_text(encoding="utf-8")

    print(f"📄 読み込み完了: {path.name} ({len(text)}文字)")
    return text


def interactive_input() -> str:
    """対話的にテキスト入力を受け付ける"""
    print("📝 企画内容をテキストで入力してください（終了: 空行でEnter2回）:")
    lines = []
    empty_count = 0
    while True:
        line = input()
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append(line)
        else:
            empty_count = 0
            lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        print("エラー: 入力が空です")
        sys.exit(1)

    print(f"📄 入力完了 ({len(text)}文字)")
    return text


def research_trends(brief: str) -> str:
    """Perplexity APIでトレンドリサーチ"""
    print("🔍 トレンドリサーチ中...")

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

    result = response.choices[0].message.content
    print("✅ トレンドリサーチ完了")
    return result


def draft_lyrics(brief: str, research: str) -> str:
    """GPT-4oで歌詞ドラフトを生成"""
    print("🎵 歌詞ドラフト生成中...")

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

    result = response.choices[0].message.content
    print("✅ 歌詞ドラフト完了")
    return result


def save_output(brief: str, research: str, lyrics: str, pdf_name: str):
    """結果をテキストファイルに保存"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    stem = Path(pdf_name).stem
    output_path = output_dir / f"{stem}_lyrics.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ULTRON - 歌詞制作ワークフロー結果\n")
        f.write("=" * 60 + "\n\n")

        f.write("■ 企画書サマリー\n")
        f.write("-" * 40 + "\n")
        f.write(brief[:500] + ("..." if len(brief) > 500 else "") + "\n\n")

        f.write("■ トレンドリサーチ\n")
        f.write("-" * 40 + "\n")
        f.write(research + "\n\n")

        f.write("■ 歌詞ドラフト\n")
        f.write("-" * 40 + "\n")
        f.write(lyrics + "\n")

    print(f"💾 保存完了: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        # 引数なし → 対話入力モード
        print("=" * 60)
        print("ULTRON - 歌詞制作ワークフロー")
        print("=" * 60)
        brief = interactive_input()
        source_name = "interactive"
    else:
        # ファイル指定モード（PDF / TXT / その他テキスト）
        brief = read_input(sys.argv[1])
        source_name = sys.argv[1]

    # Step 1: Perplexityでトレンドリサーチ
    research = research_trends(brief)

    # Step 2: GPT-4oで歌詞ドラフト
    lyrics = draft_lyrics(brief, research)

    # Step 3: 結果を保存
    output_path = save_output(brief, research, lyrics, source_name)

    print("\n" + "=" * 60)
    print("🎤 ワークフロー完了!")
    print(f"📁 結果ファイル: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
