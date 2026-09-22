"""Test path parsing logic for Windows paths."""
import sys
sys.path.insert(0, ".")

# We import only parse_doc_command — no Ollama needed
import importlib, ast, types

# Read and exec only the parse_doc_command function without full ask.py imports
import re, os

def parse_doc_command(rest):
    rest = rest.strip().strip("<>")
    if not rest:
        return "", ""
    for q in ('"', "'"):
        if rest.startswith(q):
            end_idx = rest.find(q, 1)
            if end_idx != -1:
                raw_path = rest[1:end_idx].strip()
                prompt   = rest[end_idx + 1:].strip()
                return os.path.normpath(raw_path), prompt
    tokens = rest.split()
    for end in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:end])
        norm = os.path.normpath(candidate.replace("/", os.sep))
        if os.path.isfile(norm):
            prompt = " ".join(tokens[end:]).strip()
            return norm, prompt
    first = tokens[0]
    has_drive     = len(first) >= 2 and first[1] == ":"
    has_sep       = ("/" in first or "\\" in first)
    has_extension = "." in os.path.basename(first)
    if has_drive or has_sep or has_extension:
        supported_exts = {".pdf", ".docx", ".xlsx", ".xls", ".csv",
                          ".txt", ".md", ".json", ".png", ".jpg", ".jpeg", ".webp"}
        best_path = ""
        best_end  = 0
        for end in range(1, len(tokens) + 1):
            candidate = " ".join(tokens[:end])
            ext = os.path.splitext(candidate)[1].lower()
            if ext in supported_exts:
                best_path = candidate
                best_end  = end
        if best_path:
            norm   = os.path.normpath(best_path.replace("/", os.sep))
            prompt = " ".join(tokens[best_end:]).strip()
            return norm, prompt
        norm = os.path.normpath(first.replace("/", os.sep))
        prompt = " ".join(tokens[1:]).strip()
        return norm, prompt
    return "", rest

print("Path parsing tests:")
tests = [
    (r'"D:\Users\iamsa\Downloads\report.pdf" what are the findings',
     r"D:\Users\iamsa\Downloads\report.pdf", "what are the findings"),
    (r"D:\Engineering\hackathon\data.csv",
     r"D:\Engineering\hackathon\data.csv", ""),
    (r"report.pdf tell me key points",
     "report.pdf", "tell me key points"),
    (r"D:/Users/iamsa/Downloads/data.xlsx analyze this",
     r"D:\Users\iamsa\Downloads\data.xlsx", "analyze this"),
    (r"C:\My Documents\ETP Report.pdf summarize",
     r"C:\My Documents\ETP Report.pdf", "summarize"),
]
ok = 0
for inp, exp_path, exp_prompt in tests:
    got_path, got_prompt = parse_doc_command(inp)
    path_ok   = os.path.normpath(got_path) == os.path.normpath(exp_path)
    prompt_ok = got_prompt == exp_prompt
    status = "OK" if (path_ok and prompt_ok) else "FAIL"
    if status == "OK":
        ok += 1
    print(f"  [{status}] Input: {inp[:55]!r}")
    if not path_ok:
        print(f"         Path  got={got_path!r}  expected={exp_path!r}")
    if not prompt_ok:
        print(f"         Prompt got={got_prompt!r}  expected={exp_prompt!r}")

print(f"\n{ok}/{len(tests)} path parsing tests passed.")
