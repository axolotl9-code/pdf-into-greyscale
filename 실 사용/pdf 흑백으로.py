"""
간단한 PDF 흑백 변환 스크립트(스텁).

이 파일은 저장소에 빠르게 추가할 수 있는 시작점입니다. 실제 구현에서는
- PyPDF2 / pikepdf 등으로 PDF 페이지/이미지 추출
- Pillow(OpenCV)로 이미지 흑백(그레이스케일) 변환
- 변환된 이미지를 다시 PDF로 합치기
같은 작업이 필요합니다.

사용법 (현재 스텁):
    python3 pdf_to_greyscale.py input.pdf output.pdf

현재 `convert_pdf_to_greyscale` 함수는 구현되어 있지 않으며
NotImplementedError 를 발생시킵니다. 원하시면 실제 변환 구현을 추가해 드릴게요.
"""

import argparse
from pathlib import Path


def convert_pdf_to_greyscale(input_path: Path, output_path: Path) -> None:
    """PDF를 흑백(그레이스케일)으로 변환해 저장한다.

    현재는 스텁입니다. 실제 구현은 이미지 추출/변환/재조합을 포함해야 합니다.

    Args:
        input_path: 입력 PDF 파일 경로
        output_path: 출력 PDF 파일 경로

    Raises:
        NotImplementedError: 아직 구현되지 않음
    """
    raise NotImplementedError("convert_pdf_to_greyscale 함수는 아직 구현되지 않았습니다.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert a PDF to greyscale (stub).")
    p.add_argument("input", help="Input PDF file path")
    p.add_argument("output", help="Output PDF file path")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        parser.error(f"입력 파일이 존재하지 않습니다: {input_path}")

    # 현재는 스텁임을 알림
    print(f"스텁 실행: {input_path} -> {output_path}")
    print("convert_pdf_to_greyscale 함수가 아직 구현되지 않았습니다.")

    # 실제로는 여기서 변환 함수를 호출합니다.
    # convert_pdf_to_greyscale(input_path, output_path)


if __name__ == "__main__":
    main()
