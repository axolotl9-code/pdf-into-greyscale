#!/usr/bin/env python3
"""
폴더 전체에서 PDF를 찾아 흑백(grayscale)으로 변환하여
같은 폴더에 '흑백-<원본이름>.pdf'로 저장합니다.

중복 방지 규칙:
- 이미 '흑백-'으로 시작하는 파일은 건너뜀
- 대상 폴더에 '흑백-<원본이름>.pdf'가 존재하면 건너뜀
- 같은 경로를 한 번 이상 처리하지 않도록 실경로(realpath)로 중복 체크

권장: Ghostscript가 설치되어 있으면 벡터/텍스트가 보존된 흑백 PDF 생성.
미설치 시 PyMuPDF로 래스터 흑백 PDF 생성.

사용 예:
    python3 pdf_흑백으로.py
    # 또는 특정 폴더를 지정
    python3 pdf_흑백으로.py /path/to/folder

주의: 노트북에서 쓰는 '!pip install ...' 같은 Jupyter 매직은 이 스크립트에서 제거되었습니다.
필요한 패키지는 수동으로 설치하세요: `pip install pymupdf`
"""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional
from pathlib import Path

# -------- 설정 --------
ROOT_DIR = "."  # 기본 시작 폴더 (커맨드라인 인수로 덮어쓸 수 있음)
TARGET_PREFIX = "흑백-"  # 출력 파일 접두사
TARGET_GRAYSCALE_WIDTH_PX = 4096  # PyMuPDF 래스터 변환 시 목표 가로 픽셀
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}  # 지원 이미지 확장자
IMAGES_PDF_NAME = "흑백-images.pdf"  # 이미지들을 묶은 PDF 파일명
# ----------------------


def which(cmd: str) -> Optional[str]:
    """PATH에서 실행파일 찾기 (Windows/Unix 호환)."""
    from shutil import which as _which

    return _which(cmd)


def get_gs_executable() -> Optional[str]:
    """
    Ghostscript 실행 파일 경로 탐지.
    - Linux/macOS: 'gs'
    - Windows: 'gswin64c' 또는 'gswin32c'
    """
    for cand in ("gs", "gswin64c", "gswin32c"):
        p = which(cand)
        if p:
            return p
    return None


def convert_with_ghostscript(gs: str, src: str, dst: str) -> None:
    """Ghostscript로 벡터(텍스트) 보존 흑백 변환."""
    cmd = [
        gs,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dProcessColorModel=/DeviceGray",
        "-sColorConversionStrategy=Gray",
        "-dColorConversionStrategy=/Gray",
        "-dDetectDuplicateImages",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={dst}",
        src,
    ]
    subprocess.run(cmd, check=True)


def convert_with_pymupdf(src: str, dst: str) -> None:
    """
    PyMuPDF로 페이지를 회색조 이미지로 렌더링하여 새 PDF로 저장 (래스터화).
    텍스트/벡터는 이미지가 됩니다.
    """
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("PyMuPDF가 설치되어 있지 않습니다. `pip install pymupdf` 후 다시 실행하세요.") from e

    doc = fitz.open(src)
    out = fitz.open()
    try:
        for i in range(len(doc)):
            p = doc.load_page(i)

            # 원본 페이지의 너비(포인트)를 기준으로 스케일 팩터 계산
            original_width_pts = p.rect.width
            # TARGET_GRAYSCALE_WIDTH_PX 보다 원본이 크면 1.0 유지
            scale_factor = max(1.0, TARGET_GRAYSCALE_WIDTH_PX / original_width_pts)
            matrix = fitz.Matrix(scale_factor, scale_factor)

            # 회색조 렌더링 (지정된 스케일 팩터 적용)
            pix = p.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)

            # 스케일된 이미지 크기에 맞춰 새 페이지 생성
            page = out.new_page(-1, width=pix.width, height=pix.height)
            page.insert_image(page.rect, stream=pix.tobytes("png"))
    finally:
        out.save(dst)
        out.close()
        doc.close()


def convert_images_to_pdf(image_paths: list[str], output_pdf: str) -> None:
    """
    여러 이미지를 흑백으로 변환하여 하나의 PDF로 결합.
    각 이미지는 원본 크기 그대로 한 페이지를 차지합니다 (잘리지 않음).
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Pillow가 설치되어 있지 않습니다. `pip install pillow` 후 다시 실행하세요.") from e
    
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("PyMuPDF가 설치되어 있지 않습니다. `pip install pymupdf` 후 다시 실행하세요.") from e

    pdf_doc = fitz.open()
    
    for img_path in image_paths:
        try:
            # Pillow로 이미지 열고 그레이스케일 변환
            img = Image.open(img_path)
            if img.mode != "L":
                img = img.convert("L")
            
            # 임시로 PNG 바이트로 변환 (PyMuPDF에 삽입하기 위해)
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            
            # 이미지 크기에 맞춰 PDF 페이지 생성 (72 DPI 기준, Pillow 픽셀 → 포인트)
            # 이미지가 잘리지 않도록 원본 크기 그대로 사용
            width_pt = img.width * 72 / 96  # 96 DPI 가정 (일반적 화면 DPI)
            height_pt = img.height * 72 / 96
            
            page = pdf_doc.new_page(width=width_pt, height=height_pt)
            rect = fitz.Rect(0, 0, width_pt, height_pt)
            page.insert_image(rect, stream=img_bytes.read())
            
        except Exception as e:
            print(f"[경고] 이미지 처리 실패, 건너뜀: {img_path} ({e})", file=sys.stderr)
            continue
    
    if pdf_doc.page_count > 0:
        pdf_doc.save(output_pdf)
    pdf_doc.close()


def convert_pdf_to_gray(src: str, dst: str, gs_path: Optional[str]) -> bool:
    """
    하나의 PDF를 흑백으로 변환. 성공 시 True 반환.
    gs_path가 있으면 Ghostscript 사용, 없으면 PyMuPDF 사용.
    """
    try:
        if gs_path:
            convert_with_ghostscript(gs_path, src, dst)
        else:
            convert_with_pymupdf(src, dst)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[GS 실패] {src} -> {dst} : {e}", file=sys.stderr)
    except Exception as e:
        print(f"[변환 실패] {src} -> {dst} : {e}", file=sys.stderr)
    return False


def convert_file_to_gray(src: str, dst: str, gs_path: Optional[str]) -> bool:
    """
    PDF 파일을 흑백으로 변환. 성공 시 True 반환.
    이미지는 이 함수에서 처리하지 않음 (폴더별로 일괄 처리됨).
    """
    ext = os.path.splitext(src)[1].lower()
    try:
        if ext == ".pdf":
            return convert_pdf_to_gray(src, dst, gs_path)
        else:
            return False
    except Exception as e:
        print(f"[변환 실패] {src} -> {dst} : {e}", file=sys.stderr)
        return False


def should_skip(src: str, dst: str) -> bool:
    """변환 필요 여부 판단: 이미 변환된 파일이거나 대상 존재 시 스킵."""
    base = os.path.basename(src)
    if base.startswith(TARGET_PREFIX):
        return True
    if os.path.exists(dst):
        return True
    return False


def is_supported_file(filename: str) -> bool:
    """지원하는 파일(PDF 또는 이미지) 여부 확인."""
    ext = os.path.splitext(filename)[1].lower()
    return ext == ".pdf" or ext in IMAGE_EXTENSIONS


def walk_and_convert(root: str) -> None:
    gs_path = get_gs_executable()
    if gs_path:
        print(f"[INFO] Ghostscript 발견: {gs_path} (PDF 벡터 보존 흑백 변환 사용)")
    else:
        print(f"[INFO] Ghostscript 미발견 → PyMuPDF 래스터 변환으로 진행 (목표 가로: {TARGET_GRAYSCALE_WIDTH_PX}px)")
    
    print(f"[INFO] 지원 형식: PDF (개별 변환), 이미지 (폴더별 묶어서 PDF로)")

    processed = set()  # realpath 기반 중복 방지
    total_pdf = converted_pdf = skipped_pdf = 0
    
    # 폴더별로 이미지 수집
    folder_images: dict[str, list[str]] = {}
    
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if not is_supported_file(name):
                continue
            
            src = os.path.join(dirpath, name)
            rsrc = os.path.realpath(src)
            if rsrc in processed:
                continue
            processed.add(rsrc)
            
            ext = os.path.splitext(name)[1].lower()
            
            # PDF는 개별 처리
            if ext == ".pdf":
                dst = os.path.join(dirpath, f"{TARGET_PREFIX}{name}")
                total_pdf += 1
                
                if should_skip(src, dst):
                    skipped_pdf += 1
                    print(f"[SKIP] {src}")
                    continue
                
                ok = convert_file_to_gray(src, dst, gs_path)
                if ok:
                    converted_pdf += 1
                    print(f"[OK]   {src} -> {dst}")
                    try:
                        os.remove(src)
                    except Exception as e:
                        print(f"원본 삭제 실패: {src} ({e})", file=sys.stderr)
                else:
                    print(f"[ERR]  {src}")
            
            # 이미지는 폴더별로 수집
            elif ext in IMAGE_EXTENSIONS:
                # 이미 흑백- 접두사가 있으면 스킵
                if name.startswith(TARGET_PREFIX):
                    print(f"[SKIP] {src} (이미 변환됨)")
                    continue
                
                if dirpath not in folder_images:
                    folder_images[dirpath] = []
                folder_images[dirpath].append(src)
    
    # 각 폴더별로 수집된 이미지들을 하나의 PDF로 변환
    total_img = 0
    converted_img = 0
    for folder, img_list in folder_images.items():
        if not img_list:
            continue
        
        total_img += len(img_list)
        output_pdf = os.path.join(folder, IMAGES_PDF_NAME)
        
        # 이미 존재하면 스킵
        if os.path.exists(output_pdf):
            print(f"[SKIP] 이미지 PDF가 이미 존재: {output_pdf}")
            continue
        
        try:
            print(f"[INFO] {len(img_list)}개 이미지를 PDF로 결합: {output_pdf}")
            convert_images_to_pdf(img_list, output_pdf)
            print(f"[OK]   이미지 PDF 생성: {output_pdf}")
            
            # 성공적으로 PDF 생성 후 원본 이미지 삭제
            for img_path in img_list:
                try:
                    os.remove(img_path)
                    print(f"[삭제] {img_path}")
                    converted_img += 1
                except Exception as e:
                    print(f"원본 삭제 실패: {img_path} ({e})", file=sys.stderr)
        except Exception as e:
            print(f"[ERR] 이미지 PDF 생성 실패: {output_pdf} ({e})", file=sys.stderr)

    print("\n--- 요약 ---")
    print(f"PDF: 총 {total_pdf}개 | 변환 {converted_pdf}개 | 스킵 {skipped_pdf}개")
    print(f"이미지: 총 {total_img}개 | 변환 {converted_img}개 | PDF {len(folder_images)}개 생성")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    import argparse

    p = argparse.ArgumentParser(description="폴더 내 PDF를 흑백으로 변환합니다.")
    p.add_argument("root", nargs="?", default=ROOT_DIR, help="시작 폴더 (기본: 현재 디렉터리)")
    p.add_argument("--yes", action="store_true", help="비대화형: 자동으로 진행(원본 삭제 포함/자동 설치 허용)")
    args = p.parse_args(argv)

    # 만약 Ghostscript가 없고 PyMuPDF도 없다면 설치를 제안/시도합니다.
    gs_path = get_gs_executable()
    fitz_available = True
    try:
        import fitz  # type: ignore
    except Exception:
        fitz_available = False

    if not gs_path and not fitz_available:
        # 사용자 허가 없이 자동 설치: fitz가 없고 Ghostscript가 없을 때만 설치 시도
        print("[INFO] PyMuPDF가 감지되지 않아 자동 설치를 시도합니다...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pymupdf"], check=True)
            import importlib
            importlib.invalidate_caches()
            try:
                import fitz  # type: ignore
                fitz_available = True
                print("[INFO] PyMuPDF 설치 완료")
            except Exception:
                print("[ERROR] PyMuPDF 설치 후에도 import에 실패했습니다.", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] PyMuPDF 자동 설치에 실패했습니다: {e}", file=sys.stderr)

    # Pillow 자동 설치 (이미지 변환에 필요)
    pillow_available = True
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pillow_available = False
    
    if not pillow_available:
        print("[INFO] Pillow가 감지되지 않아 자동 설치를 시도합니다...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pillow"], check=True)
            import importlib
            importlib.invalidate_caches()
            try:
                from PIL import Image  # type: ignore
                pillow_available = True
                print("[INFO] Pillow 설치 완료")
            except Exception:
                print("[ERROR] Pillow 설치 후에도 import에 실패했습니다.", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Pillow 자동 설치에 실패했습니다: {e}", file=sys.stderr)

    # 실행
    walk_and_convert(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
