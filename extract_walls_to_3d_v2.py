#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽出された壁線画像から3D座標データ(Blender形式)を生成（改良版）
壁の切れ目を最小化し、連続した壁として検出
"""
import sys
import json
import numpy as np
import cv2
from pathlib import Path

class WallTo3DConverter:
    def __init__(self, wall_height=2.4, pixel_to_meter=0.01):
        """
        Parameters:
        - wall_height: 壁の高さ（メートル）
        - pixel_to_meter: ピクセルからメートルへの変換係数
        """
        self.wall_height = wall_height
        self.pixel_to_meter = pixel_to_meter
    
    def extract_wall_lines(self, binary_img):
        """
        二値画像から壁の線分を抽出（改良版）
        """
        # 白い部分（壁）を検出
        _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 骨格化の前に軽く膨張させて隙間を埋める
        kernel = np.ones((3, 3), np.uint8)
        binary_dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 骨格化して中心線を取得
        skeleton = self._skeletonize(binary_dilated)
        
        # 線分検出（Hough変換）- より寛容なパラメータ
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi/180,
            threshold=30,        # 閾値を下げて短い線も検出
            minLineLength=10,    # 最小長を短く
            maxLineGap=30        # ギャップを大きく（連続性を保つ）
        )
        
        if lines is None:
            print("Warning: No lines detected")
            return []
        
        print(f"  Detected {len(lines)} initial line segments")
        
        # 線分を結合・整理（改良版）
        merged_lines = self._merge_lines_improved(lines)
        
        print(f"  Merged into {len(merged_lines)} wall segments")
        
        return merged_lines
    
    def _skeletonize(self, binary_img):
        """画像を骨格化（Zhang-Suen法）"""
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary_img > 0)
        return (skeleton * 255).astype(np.uint8)
    
    def _merge_lines_improved(self, lines, distance_threshold=30, angle_threshold=15):
        """
        近い線分を結合（改良版）
        - より大きな距離閾値
        - 複数回の結合パス
        """
        if len(lines) == 0:
            return []
        
        lines = lines.reshape(-1, 4)
        
        # 複数回結合を繰り返す
        for iteration in range(3):
            merged = []
            used = set()
            
            for i, line1 in enumerate(lines):
                if i in used:
                    continue
                
                x1, y1, x2, y2 = line1
                current_line = [x1, y1, x2, y2]
                used.add(i)
                
                # 近い線分を探して結合
                changed = True
                while changed:
                    changed = False
                    for j, line2 in enumerate(lines):
                        if j in used:
                            continue
                        
                        if self._should_merge(current_line, line2, distance_threshold, angle_threshold):
                            current_line = self._merge_two_lines(current_line, line2)
                            used.add(j)
                            changed = True
                
                merged.append(current_line)
            
            # 次の反復用に更新
            if len(merged) == len(lines):
                break  # これ以上結合できない
            lines = np.array(merged)
        
        return merged
    
    def _should_merge(self, line1, line2, dist_thresh, angle_thresh):
        """2本の線分が結合すべきかを判定（改良版）"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 端点間の距離を計算
        distances = [
            np.sqrt((x1-x3)**2 + (y1-y3)**2),
            np.sqrt((x1-x4)**2 + (y1-y4)**2),
            np.sqrt((x2-x3)**2 + (y2-y3)**2),
            np.sqrt((x2-x4)**2 + (y2-y4)**2),
        ]
        
        min_dist = min(distances)
        
        if min_dist > dist_thresh:
            return False
        
        # 角度の差を計算
        angle1 = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        angle2 = np.arctan2(y4-y3, x4-x3) * 180 / np.pi
        angle_diff = abs(angle1 - angle2)
        
        # 角度差を0-180度に正規化
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 平行または180度反対の場合も結合
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        return angle_diff < angle_thresh
    
    def _merge_two_lines(self, line1, line2):
        """2本の線分を結合"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # すべての端点から最も離れた2点を選ぶ
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        max_dist = 0
        best_pair = None
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (points[i], points[j])
        
        return [best_pair[0][0], best_pair[0][1], best_pair[1][0], best_pair[1][1]]
    
    def lines_to_3d_coordinates(self, lines, img_height):
        """
        2D線分を3D座標に変換（床の中心を原点とする）
        
        Returns: Blender形式のJSON
        """
        walls = []
        all_coords = []  # 座標の中心化用
        
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line
            
            # 線の長さが極端に短いものは除外
            length_px = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length_px < 5:  # 5ピクセル未満は除外
                continue
            
            # Y座標を反転（画像座標系 → Blender座標系）
            y1 = img_height - y1
            y2 = img_height - y2
            
            # ピクセル → メートル
            x1_m = x1 * self.pixel_to_meter
            y1_m = y1 * self.pixel_to_meter
            x2_m = x2 * self.pixel_to_meter
            y2_m = y2 * self.pixel_to_meter
            
            # 座標を記録（後で中心化するため）
            all_coords.append((x1_m, y1_m))
            all_coords.append((x2_m, y2_m))
            
            # 壁データ作成（まだ中心化されていない座標）
            wall = {
                "id": f"wall_{idx+1}",
                "type": "wall",
                "start": [round(x1_m, 3), round(y1_m, 3), 0.0],
                "end": [round(x2_m, 3), round(y2_m, 3), 0.0],
                "height": round(self.wall_height, 2),
                "thickness": 0.10,  # 10cm（viz_scale=100px/mで10px表示）
                "length": round(np.sqrt((x2_m-x1_m)**2 + (y2_m-y1_m)**2), 3)
            }
            
            walls.append(wall)
        
        # 床の中心を計算
        if all_coords:
            xs = [coord[0] for coord in all_coords]
            ys = [coord[1] for coord in all_coords]
            center_x = (min(xs) + max(xs)) / 2.0
            center_y = (min(ys) + max(ys)) / 2.0
            
            # すべての壁の座標を中心化
            for wall in walls:
                wall["start"][0] = round(wall["start"][0] - center_x, 3)
                wall["start"][1] = round(wall["start"][1] - center_y, 3)
                wall["end"][0] = round(wall["end"][0] - center_x, 3)
                wall["end"][1] = round(wall["end"][1] - center_y, 3)
        
        return {
            "format": "blender_floor_plan",
            "version": "1.0",
            "units": "meters",
            "metadata": {
                "wall_height": self.wall_height,
                "pixel_to_meter": self.pixel_to_meter,
                "total_walls": len(walls),
                "origin": "floor_center"
            },
            "walls": walls
        }

def process_image_to_3d(image_path, output_json, wall_height=2.4, pixel_to_meter=0.01):
    """
    壁線画像を読み込んで3D座標に変換
    """
    print(f"Processing: {image_path}")
    
    # 画像読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    print(f"  Image size: {img.shape[1]} x {img.shape[0]}")
    
    # 変換器を作成
    converter = WallTo3DConverter(wall_height, pixel_to_meter)
    
    # 壁線を抽出
    print("  Extracting wall lines...")
    lines = converter.extract_wall_lines(img)
    
    if not lines:
        print("  Error: No walls detected")
        return None
    
    # 3D座標に変換
    print("  Converting to 3D coordinates...")
    result = converter.lines_to_3d_coordinates(lines, img.shape[0])
    
    # JSON保存
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved: {output_json}")
    print(f"  Total walls: {result['metadata']['total_walls']}")
    
    # 統計情報
    total_length = sum(w['length'] for w in result['walls'])
    print(f"  Total wall length: {total_length:.2f} meters")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_walls_to_3d_v2.py <IMAGE_PATH> [output.json] [wall_height] [pixel_to_meter]")
        print()
        print("Example:")
        print("  python extract_walls_to_3d_v2.py refined_t180_w12.png walls_3d_v2.json 2.4 0.01")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "walls_3d_v2.json"
    wall_height = float(sys.argv[3]) if len(sys.argv) > 3 else 2.4
    pixel_to_meter = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    
    process_image_to_3d(image_path, output_json, wall_height, pixel_to_meter)
