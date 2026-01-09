#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refined画像を基準に分裂した壁を自動結合

使用方法:
  python auto_merge_walls.py refined.png walls_3d.json output.json
  python auto_merge_walls.py refined.png walls_3d.json output.json --radius 50 --angle 15
"""

import sys
import json
import numpy as np
import cv2
from pathlib import Path
from skimage.morphology import skeletonize
from collections import defaultdict


class WallAutoMerger:
    def __init__(self, search_radius=50, angle_tolerance=15):
        """
        Parameters:
        - search_radius: 端点から結合候補を探す半径（ピクセル）
        - angle_tolerance: 結合を許容する角度差（度）
        """
        self.search_radius = search_radius
        self.angle_tolerance = angle_tolerance
    
    def load_refined_skeleton(self, refined_image_path):
        """refined画像を読み込んで骨格化"""
        print(f"Loading refined image: {refined_image_path}")
        
        # グレースケールで読み込み
        refined = cv2.imread(str(refined_image_path), cv2.IMREAD_GRAYSCALE)
        if refined is None:
            raise ValueError(f"Cannot load image: {refined_image_path}")
        
        print(f"  Image size: {refined.shape[1]}x{refined.shape[0]}")
        
        # 二値化（黒線を白に反転）
        _, binary = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 骨格化
        print("  Skeletonizing...")
        skeleton = skeletonize(binary > 0)
        skeleton_img = (skeleton * 255).astype(np.uint8)
        
        # デバッグ用に骨格画像を保存
        skeleton_path = Path(refined_image_path).parent / "skeleton_debug.png"
        cv2.imwrite(str(skeleton_path), skeleton_img)
        print(f"  Skeleton saved: {skeleton_path}")
        
        return skeleton_img, refined.shape, refined
    
    def load_walls_json(self, json_path):
        """壁データJSONを読み込み"""
        print(f"\nLoading walls JSON: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        walls = data['walls']
        metadata = data['metadata']
        
        print(f"  Total walls: {len(walls)}")
        print(f"  Pixel to meter: {metadata.get('pixel_to_meter', 0.01)}")
        
        return data
    
    def meter_to_pixel(self, coord_m, pixel_to_meter, image_height, image_width):
        """メートル座標をピクセル座標に変換（中心化されたメートル座標を想定）"""
        x_m, y_m = coord_m[0], coord_m[1]
        # 中心化されたメートル座標を画像の中心を基準にピクセル座標に変換
        x_px = (image_width / 2) + (x_m / pixel_to_meter)
        y_px = (image_height / 2) - (y_m / pixel_to_meter)  # Y軸反転
        return (int(x_px), int(y_px))
    
    def pixel_to_meter(self, coord_px, pixel_to_meter, image_height):
        """ピクセル座標をメートル座標に変換"""
        x_px, y_px = coord_px
        x_m = x_px * pixel_to_meter
        y_m = (image_height - y_px) * pixel_to_meter
        return [round(x_m, 3), round(y_m, 3), 0.0]
    
    def has_skeleton_continuation(self, skeleton, point_px, radius):
        """指定点から骨格線が続いているか確認"""
        x, y = point_px
        h, w = skeleton.shape
        
        # 範囲外チェック
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        
        # 指定半径内の骨格ピクセル数をカウント
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        
        roi = skeleton[y_min:y_max, x_min:x_max]
        skeleton_pixels = np.sum(roi > 0)
        
        # 骨格線が一定以上存在すれば「続いている」と判定（閾値を下げる）
        return skeleton_pixels > 2  # 5→2に変更してより寛容に
    
    def get_skeleton_direction(self, skeleton, point_px, radius):
        """指定点付近の骨格線の方向を取得（角度、度）"""
        x, y = point_px
        h, w = skeleton.shape
        
        # ROI抽出
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        
        roi = skeleton[y_min:y_max, x_min:x_max]
        
        # 骨格ピクセルの座標を取得
        ys, xs = np.where(roi > 0)
        
        if len(xs) < 2:
            return None
        
        # 主成分分析で方向を推定（簡易版：最小二乗法）
        xs_global = xs + x_min
        ys_global = ys + y_min
        
        # 中心化
        x_mean = np.mean(xs_global)
        y_mean = np.mean(ys_global)
        
        # 共分散行列
        cov = np.cov(xs_global - x_mean, ys_global - y_mean)
        
        # 固有値・固有ベクトル
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # 最大固有値の固有ベクトルが主方向
        max_idx = np.argmax(eigenvalues)
        direction_vec = eigenvectors[:, max_idx]
        
        # 角度に変換
        angle = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        
        return angle
    
    def calculate_wall_angle(self, wall_start, wall_end):
        """壁の角度を計算（度）"""
        dx = wall_end[0] - wall_start[0]
        dy = wall_end[1] - wall_start[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle
    
    def angle_difference(self, angle1, angle2):
        """2つの角度の差（0-90度、反対方向も考慮）"""
        diff = abs(angle1 - angle2) % 360
        if diff > 180:
            diff = 360 - diff
        
        # 平行または180度反対の場合も許容（90度以上なら180度から差を引く）
        if diff > 90:
            diff = 180 - diff
        
        return diff
    
    def find_merge_candidates(self, walls, skeleton, pixel_to_meter, image_shape):
        """結合候補のペアを探索"""
        print("\nSearching for merge candidates...")
        
        image_height = image_shape[0]
        image_width = image_shape[1]
        merge_pairs = []
        
        # 各壁の端点をピクセル座標に変換
        wall_endpoints = []
        for i, wall in enumerate(walls):
            start_px = self.meter_to_pixel(wall['start'], pixel_to_meter, image_height, image_width)
            end_px = self.meter_to_pixel(wall['end'], pixel_to_meter, image_height, image_width)
            wall_angle = self.calculate_wall_angle(start_px, end_px)
            
            wall_endpoints.append({
                'wall_id': i,
                'start_px': start_px,
                'end_px': end_px,
                'angle': wall_angle
            })
            
            # デバッグ出力（最初の3つだけ）
            if i < 3:
                print(f"  Wall {i}: start={start_px}, end={end_px}, angle={wall_angle:.1f}°")
        
        # 各壁ペアについて、最も近い端点の組み合わせで判定
        checked_pairs = set()
        near_pairs_count = 0
        skeleton_fail_count = 0
        angle_fail_count = 0
        
        for i, wall_i in enumerate(wall_endpoints):
            for j, wall_j in enumerate(wall_endpoints):
                if i >= j:  # 重複チェックを避ける
                    continue
                
                pair_key = (i, j)
                if pair_key in checked_pairs:
                    continue
                
                # 各壁の4つの端点の組み合わせから最短距離を見つける
                endpoints_i = [wall_i['start_px'], wall_i['end_px']]
                endpoints_j = [wall_j['start_px'], wall_j['end_px']]
                
                min_dist = float('inf')
                closest_pair = None
                
                for ep_i in endpoints_i:
                    for ep_j in endpoints_j:
                        dist = np.sqrt(
                            (ep_i[0] - ep_j[0])**2 + 
                            (ep_i[1] - ep_j[1])**2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (ep_i, ep_j)
                
                # 最短距離が検索範囲内かチェック
                if min_dist > self.search_radius:
                    continue
                
                near_pairs_count += 1
                
                # この近接端点の周辺に骨格線があるかチェック
                mid_x = (closest_pair[0][0] + closest_pair[1][0]) // 2
                mid_y = (closest_pair[0][1] + closest_pair[1][1]) // 2
                
                if not self.has_skeleton_continuation(skeleton, (mid_x, mid_y), int(min_dist) + 10):
                    skeleton_fail_count += 1
                    continue
                
                # 角度差をチェック
                angle_diff = self.angle_difference(wall_i['angle'], wall_j['angle'])
                
                if angle_diff > self.angle_tolerance:
                    angle_fail_count += 1
                    continue
                
                # 結合候補として追加
                merge_pairs.append(pair_key)
                checked_pairs.add(pair_key)
                print(f"  Found merge candidate: Wall {i} <-> Wall {j} (distance={min_dist:.1f}px, angle_diff={angle_diff:.1f}°)")
        
        print(f"  Near pairs (dist<{self.search_radius}px): {near_pairs_count}")
        print(f"  Rejected by skeleton check: {skeleton_fail_count}")
        print(f"  Rejected by angle check: {angle_fail_count}")
        print(f"  Total merge candidates found: {len(merge_pairs)}")
        return merge_pairs
    
    def merge_walls(self, walls, merge_pairs):
        """壁を結合（グラフの連結成分を統合）"""
        print(f"\nMerging {len(merge_pairs)} wall pairs...")
        
        if len(merge_pairs) == 0:
            print("  No walls to merge.")
            return walls
        
        # グラフ構造を構築（Union-Find）
        parent = {i: i for i in range(len(walls))}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 結合ペアをグラフに追加
        for i, j in merge_pairs:
            union(i, j)
        
        # 連結成分ごとにグループ化
        groups = defaultdict(list)
        for i in range(len(walls)):
            root = find(i)
            groups[root].append(i)
        
        # 各グループを1つの壁に統合
        merged_walls = []
        merged_count = 0
        
        for root, wall_ids in groups.items():
            if len(wall_ids) == 1:
                # 単独の壁はそのまま
                merged_walls.append(walls[wall_ids[0]])
            else:
                # 複数の壁を結合
                print(f"  Merging walls: {wall_ids}")
                merged_wall = self.merge_wall_group(walls, wall_ids)
                merged_walls.append(merged_wall)
                merged_count += len(wall_ids) - 1
        
        print(f"  Merged {merged_count} walls. Total: {len(walls)} -> {len(merged_walls)}")
        
        return merged_walls
    
    def merge_wall_group(self, walls, wall_ids):
        """複数の壁を1つに統合"""
        # 全端点を収集
        all_points = []
        for wid in wall_ids:
            wall = walls[wid]
            all_points.append(wall['start'])
            all_points.append(wall['end'])
        
        # 端点の端点を見つける（最も離れた2点）
        max_dist = 0
        best_pair = (all_points[0], all_points[1])
        
        for i, p1 in enumerate(all_points):
            for j, p2 in enumerate(all_points):
                if i >= j:
                    continue
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (p1, p2)
        
        # 統合された壁を作成
        merged_wall = {
            'id': walls[wall_ids[0]]['id'],  # 最初の壁のIDを継承
            'start': best_pair[0],
            'end': best_pair[1],
            'height': walls[wall_ids[0]]['height'],
            'thickness': walls[wall_ids[0]]['thickness'],
            'length': round(max_dist, 3)
        }
        
        return merged_wall
    
    def visualize_debug(self, refined_img, walls, pixel_to_meter, image_shape, merge_pairs, output_dir):
        """デバッグ用：壁の端点を可視化"""
        print("\nGenerating debug visualization...")
        
        # カラー画像に変換
        if len(refined_img.shape) == 2:
            vis_img = cv2.cvtColor(refined_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = refined_img.copy()
        
        image_height, image_width = image_shape[0], image_shape[1]
        
        # 各壁の端点を描画
        for i, wall in enumerate(walls):
            start_px = self.meter_to_pixel(wall['start'], pixel_to_meter, image_height, image_width)
            end_px = self.meter_to_pixel(wall['end'], pixel_to_meter, image_height, image_width)
            
            # 壁の線を描画（青）
            cv2.line(vis_img, start_px, end_px, (255, 0, 0), 2)
            
            # 端点を描画（緑の円）
            cv2.circle(vis_img, start_px, 8, (0, 255, 0), -1)
            cv2.circle(vis_img, end_px, 8, (0, 255, 0), -1)
            
            # 壁IDを表示
            mid_x = (start_px[0] + end_px[0]) // 2
            mid_y = (start_px[1] + end_px[1]) // 2
            cv2.putText(vis_img, str(i), (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 結合候補を強調表示（赤線）
        for i, j in merge_pairs:
            wall_i = walls[i]
            wall_j = walls[j]
            
            # 最も近い端点ペアを見つける
            endpoints_i = [
                self.meter_to_pixel(wall_i['start'], pixel_to_meter, image_height, image_width),
                self.meter_to_pixel(wall_i['end'], pixel_to_meter, image_height, image_width)
            ]
            endpoints_j = [
                self.meter_to_pixel(wall_j['start'], pixel_to_meter, image_height, image_width),
                self.meter_to_pixel(wall_j['end'], pixel_to_meter, image_height, image_width)
            ]
            
            min_dist = float('inf')
            closest_pair = None
            
            for ep_i in endpoints_i:
                for ep_j in endpoints_j:
                    dist = np.sqrt((ep_i[0]-ep_j[0])**2 + (ep_i[1]-ep_j[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (ep_i, ep_j)
            
            if closest_pair:
                cv2.line(vis_img, closest_pair[0], closest_pair[1], (0, 0, 255), 3)
                cv2.circle(vis_img, closest_pair[0], 12, (0, 255, 255), 2)
                cv2.circle(vis_img, closest_pair[1], 12, (0, 255, 255), 2)
        
        # 保存
        debug_path = output_dir / "merge_debug_visualization.png"
        cv2.imwrite(str(debug_path), vis_img)
        print(f"  Debug visualization saved: {debug_path}")
    
    def process(self, refined_image_path, walls_json_path, output_json_path):
        """メイン処理"""
        print("="*60)
        print("Auto Wall Merger")
        print(f"  Search radius: {self.search_radius}px")
        print(f"  Angle tolerance: {self.angle_tolerance}°")
        print("="*60)
        
        # 1. refined画像を骨格化
        skeleton, image_shape, refined_original = self.load_refined_skeleton(refined_image_path)
        
        # 2. 壁データJSONを読み込み
        data = self.load_walls_json(walls_json_path)
        walls = data['walls']
        pixel_to_meter = data['metadata'].get('pixel_to_meter', 0.01)
        
        # 3. 結合候補を探索
        merge_pairs = self.find_merge_candidates(walls, skeleton, pixel_to_meter, image_shape)
        
        # デバッグ: 壁の端点を可視化
        self.visualize_debug(refined_original, walls, pixel_to_meter, image_shape, 
                            merge_pairs, Path(refined_image_path).parent)
        
        # 4. 壁を結合
        merged_walls = self.merge_walls(walls, merge_pairs)
        
        # 5. 結果を保存
        data['walls'] = merged_walls
        data['metadata']['total_walls'] = len(merged_walls)
        
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Saved: {output_path}")
        print(f"Original walls: {len(walls)}")
        print(f"Merged walls: {len(merged_walls)}")
        print(f"Reduction: {len(walls) - len(merged_walls)} walls")
        print(f"{'='*60}")
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto merge split walls using refined image as reference')
    parser.add_argument('refined_image', help='Path to refined image (PNG)')
    parser.add_argument('walls_json', help='Path to walls JSON file')
    parser.add_argument('output_json', help='Path to output merged JSON file')
    parser.add_argument('--radius', type=int, default=50, help='Search radius in pixels (default: 50)')
    parser.add_argument('--angle', type=int, default=15, help='Angle tolerance in degrees (default: 15)')
    
    args = parser.parse_args()
    
    merger = WallAutoMerger(search_radius=args.radius, angle_tolerance=args.angle)
    merger.process(args.refined_image, args.walls_json, args.output_json)


if __name__ == "__main__":
    main()
