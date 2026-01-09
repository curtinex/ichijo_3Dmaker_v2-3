#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D座標データを2D平面図として可視化
"""
import json
import numpy as np
import cv2
import sys

def visualize_3d_walls(json_path, output_image="walls_3d_visualization.png", scale=50, highlight_wall_ids=None):
    """
    3D座標JSONを読み込んで2D平面図として描画
    
    Parameters:
    - scale: メートルからピクセルへの変換スケール（大きいほど拡大）
    - highlight_wall_ids: 赤色で強調表示する壁のIDリスト
    """
    print(f"Loading: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    walls = data['walls']
    print(f"Total walls: {len(walls)}")
    
    # すべての座標から画像サイズを決定
    all_x = []
    all_y = []
    
    for wall in walls:
        all_x.extend([wall['start'][0], wall['end'][0]])
        all_y.extend([wall['start'][1], wall['end'][1]])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # 画像サイズを計算（余白込み）
    margin = 50
    img_width = int((max_x - min_x) * scale) + 2 * margin
    img_height = int((max_y - min_y) * scale) + 2 * margin
    
    print(f"Image size: {img_width} x {img_height}")
    print(f"Floor plan bounds: {min_x:.2f} to {max_x:.2f} m (X), {min_y:.2f} to {max_y:.2f} m (Y)")
    
    # 白背景の画像を作成
    canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # 壁を描画
    for wall in walls:
        # メートル → ピクセル座標
        x1 = int((wall['start'][0] - min_x) * scale) + margin
        y1 = int((wall['start'][1] - min_y) * scale) + margin
        x2 = int((wall['end'][0] - min_x) * scale) + margin
        y2 = int((wall['end'][1] - min_y) * scale) + margin
        
        # Y軸反転（画像座標系）
        y1 = img_height - y1
        y2 = img_height - y2
        
        # 壁の太さを計算
        thickness_px = max(2, int(wall['thickness'] * scale))
        
        # 強調表示する壁は赤色、それ以外は黒
        wall_id = wall.get('id')
        if highlight_wall_ids and wall_id in highlight_wall_ids:
            color = (0, 0, 255)  # 赤色 (BGR)
            thickness_px = max(3, thickness_px + 1)  # 少し太く
        else:
            color = (0, 0, 0)  # 黒色
        
        # 線を描画
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness_px)
    
    # グリッド線を描画（0.45mごと = 45cm）
    grid_color = (200, 200, 200)
    grid_spacing = 0.45  # visualizationの1マス = 45cm（一条工務店の図面の1マス相当）
    for x_m in np.arange(0, max_x - min_x + 1, grid_spacing):
        x_px = int(x_m * scale) + margin
        cv2.line(canvas, (x_px, 0), (x_px, img_height), grid_color, 1)
    
    for y_m in np.arange(0, max_y - min_y + 1, grid_spacing):
        y_px = int(y_m * scale) + margin
        y_px_inv = img_height - y_px
        cv2.line(canvas, (0, y_px_inv), (img_width, y_px_inv), grid_color, 1)
    
    # スケール表示を追加
    scale_text = f"Scale: 1m = {scale}px | grid: 45cm | Total: {len(walls)} walls"
    cv2.putText(canvas, scale_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 保存
    cv2.imwrite(output_image, canvas)
    print(f"Saved: {output_image}")
    
    return canvas

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_3d_walls.py <JSON_PATH> [output.png] [scale]")
        print()
        print("Example:")
        print("  python visualize_3d_walls.py walls_3d.json walls_viz.png 50")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "walls_3d_visualization.png"
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    visualize_3d_walls(json_path, output_image, scale)
