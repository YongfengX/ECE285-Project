import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract raw text from a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    doc.close()
    full_text = "\n".join(pages)

    # Basic cleanup
    full_text = full_text.replace("\u00a0", " ")
    full_text = re.sub(r"[ \t]+\n", "\n", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return full_text.strip()


def normalize_text(text: str) -> str:
    """
    Light normalization for parsing robustness.
    """
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("￾", "")
    text = re.sub(r"Page\s+\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_problem_blocks_hw(text: str) -> List[Tuple[str, str]]:
    """
    Split homework text into problem-level blocks based on patterns like:
    1. Problem 1: Fields ...
    2. Problem 2: Field Operations ...
    """
    pattern = re.compile(
        r"(?ms)^\s*(\d+)\.\s*Problem\s+\d+\s*:\s*(.*?)(?=^\s*\d+\.\s*Problem\s+\d+\s*:|\Z)"
    )
    matches = pattern.findall(text)
    blocks = []
    for problem_num, body in matches:
        blocks.append((problem_num, body.strip()))
    return blocks


def find_problem_blocks_solution(text: str) -> List[Tuple[str, str]]:
    """
    Split solution text into problem-level blocks based on patterns like:
    1. (a) ...
    2. (a) ...
    3. (a) ...
    """
    pattern = re.compile(
        r"(?ms)^\s*(\d+)\.\s*(.*?)(?=^\s*\d+\.\s*|\Z)"
    )
    matches = pattern.findall(text)
    blocks = []
    for problem_num, body in matches:
        blocks.append((problem_num, body.strip()))
    return blocks


def split_subparts(block_text: str) -> Dict[str, str]:
    """
    Split a problem block into subparts (a), (b), (c), ...
    Returns dict like {"a": "...", "b": "..."}.
    If no subparts are found, returns {"main": block_text}.
    """
    # Match lines or inline markers like (a), (b), ...
    pattern = re.compile(r"(?ms)(?:^|\n)\s*\(([a-z])\)\s*(.*?)(?=(?:\n\s*\([a-z]\)\s*)|\Z)")
    matches = pattern.findall(block_text)

    if not matches:
        return {"main": block_text.strip()}

    subparts = {}
    for letter, content in matches:
        subparts[letter] = content.strip()
    return subparts


def extract_problem_title(hw_block: str) -> Tuple[str, str]:
    """
    Try to separate the title from the body.
    Example:
      'Fields Are the following sets fields?...'
    returns:
      ('Fields', 'Are the following sets fields?...')
    """
    # crude but works well for this homework style
    first_line = hw_block.split("\n", 1)[0].strip()
    m = re.match(r"^([A-Za-z0-9 ,&/\-]+?)(?:\.\s+|\s{2,}|$)(.*)", first_line)
    if m:
        title = m.group(1).strip()
        remainder = hw_block.strip()
        return title, remainder
    return "", hw_block.strip()


def build_chunks(
    hw_pdf: str,
    sol_pdf: str,
    course: str = "ECE269",
    hw_id: str = "HW2"
) -> List[dict]:
    hw_text = normalize_text(extract_pdf_text(hw_pdf))
    sol_text = normalize_text(extract_pdf_text(sol_pdf))

    hw_blocks = find_problem_blocks_hw(hw_text)
    sol_blocks = find_problem_blocks_solution(sol_text)

    hw_map = {pid: block for pid, block in hw_blocks}
    sol_map = {pid: block for pid, block in sol_blocks}

    chunks = []

    all_problem_ids = sorted(set(hw_map.keys()) | set(sol_map.keys()), key=lambda x: int(x))

    for pid in all_problem_ids:
        hw_block = hw_map.get(pid, "")
        sol_block = sol_map.get(pid, "")

        title, hw_body = extract_problem_title(hw_block) if hw_block else ("", "")
        hw_subparts = split_subparts(hw_body) if hw_body else {}
        sol_subparts = split_subparts(sol_block) if sol_block else {}

        # If neither has subparts, make a single chunk
        sub_ids = sorted(set(hw_subparts.keys()) | set(sol_subparts.keys()))
        if not sub_ids:
            sub_ids = ["main"]

        for sid in sub_ids:
            problem_text = hw_subparts.get(sid, "").strip()
            solution_text = sol_subparts.get(sid, "").strip()

            # Skip empty pair
            if not problem_text and not solution_text:
                continue

            base_id = f"{course.lower()}_{hw_id.lower()}_q{pid}{sid if sid != 'main' else ''}"

            if problem_text:
                chunks.append({
                    "chunk_id": f"{base_id}_problem",
                    "course": course,
                    "assignment": hw_id,
                    "problem_id": pid,
                    "sub_id": sid,
                    "chunk_type": "problem",
                    "title": title,
                    "source_file": Path(hw_pdf).name,
                    "text": problem_text
                })

            if solution_text:
                chunks.append({
                    "chunk_id": f"{base_id}_solution",
                    "course": course,
                    "assignment": hw_id,
                    "problem_id": pid,
                    "sub_id": sid,
                    "chunk_type": "solution",
                    "title": title,
                    "source_file": Path(sol_pdf).name,
                    "text": solution_text
                })

            # Paired QA chunk for retrieval convenience
            if problem_text and solution_text:
                chunks.append({
                    "chunk_id": f"{base_id}_qa",
                    "course": course,
                    "assignment": hw_id,
                    "problem_id": pid,
                    "sub_id": sid,
                    "chunk_type": "qa_pair",
                    "title": title,
                    "source_file": f"{Path(hw_pdf).name} + {Path(sol_pdf).name}",
                    "text": (
                        f"Problem {pid}"
                        f"{'(' + sid + ')' if sid != 'main' else ''}:\n"
                        f"{problem_text}\n\n"
                        f"Solution:\n{solution_text}"
                    )
                })

    return chunks


def save_jsonl(chunks: List[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    hw_pdf = r"WI2026_HW 2.pdf"
    sol_pdf = r"Wi2026_HW2_Solution.pdf"
    out_path = r"ece269_hw2_chunks.jsonl"

    chunks = build_chunks(
        hw_pdf=hw_pdf,
        sol_pdf=sol_pdf,
        course="ECE269",
        hw_id="HW2"
    )
    save_jsonl(chunks, out_path)

    print(f"Saved {len(chunks)} chunks to: {out_path}")
    for c in chunks[:10]:
        print("-" * 80)
        print(c["chunk_id"])
        print(c["chunk_type"])
        print(c["text"][:300], "...")


if __name__ == "__main__":
    main()