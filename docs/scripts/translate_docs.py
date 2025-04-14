# ruff: noqa
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("openai").setLevel(logging.DEBUG)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Define the source and target directories
source_dir = "docs"
languages = {
    "ja": "Japanese",
    # Add more languages here, e.g., "fr": "French"
}

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define dictionaries for translation control
do_not_translate = [
    "OpenAI",
    "Agents SDK",
    "Hello World",
    "Model context protocol",
    "structured outputs",
    "Chain-of-Thought",
    "Chat Completions",
    "Computer-Using Agent",
    "Code Interpreter",
    "Function Calling",
    "LLM",
    "Operator",
    "Playground",
    "Realtime API",
    "Sora",
    # Add more terms here
]

eng_to_non_eng_mapping = {
    "ja": {
        "agents": "エージェント",
        "computer use": "コンピュータ操作",
        "OAI hosted tools": "OpenAI がホストするツール",
        "well formed data": "適切な形式のデータ",
        "guardrail": "ガードレール",
        "handoffs": "ハンドオフ",
        "function tools": "関数ツール",
        "tracing": "トレーシング",
        "code examples": "コード例",
        "vector store": "ベクトルストア",
        "deep research": "ディープリサーチ",
        "category": "カテゴリー",
        "user": "ユーザー",
        "parameter": "パラメーター",
        "processor": "プロセッサー",
        "server": "サーバー",
        "web search": "Web 検索",
        "file search": "ファイル検索",
        "streaming": "ストリーミング",
        "system prompt": "システムプロンプト",
        "Python first": "Python ファースト",
        # Add more Japanese mappings here
    },
    # Add more languages here
}
eng_to_non_eng_instructions = {
    "common": [
        "* The term 'examples' must be code examples when the page mentions the code examples in the repo, it can be translated as either 'code examples' or 'sample code'.",
        "* The term 'primitives' can be translated as basic components.",
        "* When the terms 'instructions' and 'tools' are mentioned as API parameter names, they must be kept as is.",
        "* The terms 'temperature', 'top_p', 'max_tokens', 'presence_penalty', 'frequency_penalty' as parameter names must be kept as is.",
    ],
    "ja": [
        "* The term 'result' in the Runner guide context must be translated like 'execution results'",
        "* The term 'raw' in 'raw response events' must be kept as is",
        "* You must consistently use polite wording such as です/ます rather than である/なのだ.",
        # Add more Japanese mappings here
    ],
    # Add more languages here
}


def built_instructions(target_language: str, lang_code: str) -> str:
    do_not_translate_terms = "\n".join(do_not_translate)
    specific_terms = "\n".join(
        [f"* {k} -> {v}" for k, v in eng_to_non_eng_mapping.get(lang_code, {}).items()]
    )
    specific_instructions = "\n".join(
        eng_to_non_eng_instructions.get("common", [])
        + eng_to_non_eng_instructions.get(lang_code, [])
    )
    return f"""You are an expert technical translator.

Your task: translate the markdown passed as a user input from English into {target_language}.  

############################
##  OUTPUT REQUIREMENTS  ##
############################
- Return **only** the translated markdown, with the original markdown structure preserved.
- Do **not** add explanations, comments, or metadata.

#########################
##  GENERAL RULES      ##
#########################
- The output quality must be great enough to be used for public documentation.
- Be professional and polite.
- Keep the tone **natural** and concise.
- Do not omit any content. If a segment should stay in English, copy it verbatim.
- Do not change the markdown data structure, including the indentations.
- Keep all placeholders such as `CODE_BLOCK_*` and `CODE_LINE_PREFIX` unchanged.
- Convert asset paths: `./assets/…` → `../assets/…`.  
  *Example:* `![img](./assets/pic.png)` → `![img](../assets/pic.png)`
- Treat the **Do‑Not‑Translate list** and **Term‑Specific list** as case‑insensitive; preserve the original casing you see.
- Skip translation for:
  - Inline code surrounded by single back‑ticks ( `like_this` ).
  - Fenced code blocks delimited by ``` or ~~~, including all comments inside them.
  - Link URLs inside `[label](URL)` – translate the label, never the URL.

#########################
##  LANGUAGE‑SPECIFIC  ##
#########################
*(applies only when {target_language} = Japanese)*  
- Insert a half‑width space before and after all alphanumeric terms.  
- Add a half‑width space just outside markdown emphasis markers: ` **太字** ` (good) vs `** 太字 **` (bad).

#########################
##  DO NOT TRANSLATE   ##
#########################
When replacing the following terms, do not have extra spaces before/after them:
{do_not_translate_terms}

#########################
##  TERM‑SPECIFIC      ##
#########################
Translate these terms exactly as provided (no extra spaces):  
{specific_terms}

#########################
##  EXTRA GUIDELINES   ##
#########################
{specific_instructions}

#########################
##  IF UNSURE          ##
#########################
If you are uncertain about a term, leave the original English term in parentheses after your translation.

#########################
##  FINAL REMINDER     ##
#########################
Return **only** the translated markdown text. No extra commentary.
"""


# Function to translate and save files
def translate_file(file_path: str, target_path: str, lang_code: str) -> None:
    print(f"Translating {file_path} into a different language: {lang_code}")
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Split content into lines
    lines: list[str] = content.splitlines()
    chunks: list[str] = []
    current_chunk: list[str] = []

    # Split content into chunks of up to 120 lines, ensuring splits occur before section titles
    in_code_block = False
    code_blocks: list[str] = []
    code_block_chunks: list[str] = []
    for line in lines:
        if len(current_chunk) >= 120 and not in_code_block and line.startswith("#"):
            chunks.append("\n".join(current_chunk))
            current_chunk = []
        if line.strip().startswith("```"):
            code_block_chunks.append(line)
            if in_code_block is True:
                code_blocks.append("\n".join(code_block_chunks))
                current_chunk.append(f"CODE_BLOCK_{(len(code_blocks) - 1):02}")
                code_block_chunks.clear()
            in_code_block = not in_code_block
            continue
        if in_code_block is True:
            code_block_chunks.append(line)
        else:
            current_chunk.append(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # Translate each chunk separately and combine results
    translated_content: list[str] = []
    for chunk in chunks:
        instructions = built_instructions(languages[lang_code], lang_code)
        if OPENAI_MODEL.startswith("o"):
            response = openai_client.responses.create(
                model=OPENAI_MODEL,
                instructions=instructions,
                input=chunk,
            )
            translated_content.append(response.output_text)
        else:
            response = openai_client.responses.create(
                model=OPENAI_MODEL,
                instructions=instructions,
                input=chunk,
                temperature=0.0,
            )
            translated_content.append(response.output_text)

    translated_text = "\n".join(translated_content)
    for idx, code_block in enumerate(code_blocks):
        translated_text = translated_text.replace(f"CODE_BLOCK_{idx:02}", code_block)

    # Save the combined translated content
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(translated_text)


def translate_single_source_file(file_path: str) -> None:
    relative_path = os.path.relpath(file_path, source_dir)
    if "ref/" in relative_path or not file_path.endswith(".md"):
        return

    for lang_code in languages:
        target_dir = os.path.join(source_dir, lang_code)
        target_path = os.path.join(target_dir, relative_path)

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Translate and save the file
        translate_file(file_path, target_path, lang_code)


def main():
    # Traverse the source directory
    for root, _, file_names in os.walk(source_dir):
        # Skip the target directories
        if any(lang in root for lang in languages):
            continue
        # Increasing this will make the translation faster; you can decide considering the model's capacity
        concurrency = 6
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for file_name in file_names:
                filepath = os.path.join(root, file_name)
                futures.append(executor.submit(translate_single_source_file, filepath))
                if len(futures) >= concurrency:
                    for future in futures:
                        future.result()
                    futures.clear()

    print("Translation completed.")


if __name__ == "__main__":
    # translate_single_source_file("docs/index.md")
    main()
