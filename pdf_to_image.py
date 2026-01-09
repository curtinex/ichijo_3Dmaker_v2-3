"""
PDFファイルを画像に変換するユーティリティ（PyMuPDF使用）
"""

import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image


def pdf_to_image(pdf_path: str, output_path: str = None, page_number: int = 0, dpi: int = 300) -> str:
    """
    PDFファイルを画像に変換する
    
    Args:
        pdf_path: PDFファイルのパス
        output_path: 出力画像のパス（Noneの場合は自動生成）
        page_number: 変換するページ番号（0始まり）
        dpi: 解像度（デフォルト300dpi）
        
    Returns:
        str: 出力画像のパス
    """
    print(f"PDFを読み込み中: {pdf_path}")
    
    try:
        # PDFを開く
        doc = fitz.open(pdf_path)
        
        if page_number >= len(doc):
            raise ValueError(f"ページ番号 {page_number} は範囲外です（総ページ数: {len(doc)}）")
        
        # ページを取得
        page = doc[page_number]
        
        # 拡大率を計算（DPIに基づく）
        zoom = dpi / 72  # PDFは72dpiがベース
        mat = fitz.Matrix(zoom, zoom)
        
        # ページを画像に変換
        pix = page.get_pixmap(matrix=mat)
        
        # 出力パスを生成
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = f"{base_name}_page{page_number}.png"
        
        # 画像を保存
        pix.save(output_path)
        
        doc.close()
        
        print(f"画像を保存しました: {output_path}")
        print(f"サイズ: {pix.width} x {pix.height}")
        
        return output_path
        
    except Exception as e:
        print(f"エラー: PDFの変換に失敗しました - {e}")
        raise


def pdf_to_opencv_image(pdf_path: str, page_number: int = 0, dpi: int = 300) -> np.ndarray:
    """
    PDFファイルをOpenCV形式の画像に変換する
    
    Args:
        pdf_path: PDFファイルのパス
        page_number: 変換するページ番号（0始まり）
        dpi: 解像度（デフォルト300dpi）
        
    Returns:
        np.ndarray: OpenCV形式の画像（BGR）
    """
    print(f"PDFを読み込み中: {pdf_path} (ページ {page_number + 1})")
    
    try:
        # PDFを開く
        doc = fitz.open(pdf_path)
        
        if page_number >= len(doc):
            raise ValueError(f"ページ番号 {page_number} は範囲外です（総ページ数: {len(doc)}）")
        
        # ページを取得
        page = doc[page_number]
        
        # 拡大率を計算（DPIに基づく）
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # ページを画像に変換
        pix = page.get_pixmap(matrix=mat)
        
        # numpy配列に変換
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # RGBまたはRGBAからBGRに変換
        if pix.n == 4:  # RGBA
            opencv_image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            opencv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        else:
            opencv_image = img_data
        
        doc.close()
        
        print(f"画像を読み込みました: {opencv_image.shape}")
        
        return opencv_image
        
    except Exception as e:
        print(f"エラー: PDFの変換に失敗しました - {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python pdf_to_image.py <PDFファイルのパス> [ページ番号] [出力パス]")
        print("例: python pdf_to_image.py test.pdf 0 output.png")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        result_path = pdf_to_image(pdf_path, output_path, page_number)
        print(f"\n変換完了: {result_path}")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)
