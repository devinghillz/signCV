"""
0_DataPreprocessing.py 출력을 run_signcv.sh가 읽을 수 있는
단일 파일로 합칩니다.

실행:
    python combine_bbq_data.py
출력:
    processed_bbq/combined_forget_set.json  (axis 필드 포함)
"""
import json
from pathlib import Path

# 0_DataPreprocessing.py 카테고리 → run_signcv.sh axis 이름 매핑
AXIS_MAP = {
    'Race_ethnicity': 'race',
    'SES':            'ses',
    'Gender_identity':'gender',
}

def main():
    combined = []
    for category, axis in AXIS_MAP.items():
        path = Path(f"processed_bbq/{category}/forget_set.json")
        if not path.exists():
            print(f"[!] 없음: {path} — 0_DataPreprocessing.py를 먼저 실행하세요")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            item["axis"] = axis   # run_signcv.sh --axis_field axis 필터링용
        combined.extend(data)
        print(f"[+] {category}: {len(data)} samples (axis='{axis}')")

    out_path = Path("processed_bbq/combined_forget_set.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(combined)} samples → {out_path}")

if __name__ == "__main__":
    main()
