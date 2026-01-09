#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlitアプリ: 図面(PDF/JPG/PNG) → 壁線抽出 → 3D(JSON) → Blender用スクリプト生成

使い方:
  1) 下のコマンドで起動
     streamlit run streamlit_app.py
  2) PDF/画像をアップロード → パラメータを調整 → [変換を実行]
  3) 生成されたJSON/Blenderスクリプト/可視化をダウンロード
"""

import io
import re
import time
import json
import math
from pathlib import Path
from datetime import datetime
import zipfile

import numpy as np
import streamlit as st
import fitz  # PyMuPDF (for page count)
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# ローカルモジュール
from pdf_to_image import pdf_to_image
from refine_from_image import refine_floor_plan_from_image
from extract_walls_to_3d_v2 import process_image_to_3d
from visualize_3d_walls import visualize_3d_walls
from auto_merge_walls import WallAutoMerger

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def _save_uploaded_file(uploaded_file, dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst_path


def _generate_3d_viewer_html(json_path: Path, out_path: Path, with_lights: bool = False) -> Path:
    """Three.js HTMLビューアを自動生成（JSON内容を直接埋め込み）
    
    Args:
        json_path: JSONファイルのパス
        out_path: 出力HTMLのパス
        with_lights: True の場合、天井とスポットライトを表示
    """
    # JSONファイルを読み込んで内容を埋め込む
    with open(json_path, 'r', encoding='utf-8') as f:
        json_content = f.read()
    
    html_template = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>間取り図 3Dビューア</title>
    <style>
        body { margin: 0; overflow: hidden; font-family: sans-serif; }
        #container { width: 100vw; height: 100vh; }
        #info {
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 10px; border-radius: 5px; font-size: 14px;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="info">
        <strong>間取り図 3Dビューア</strong><br>
        初期化中...
    </div>
    <div id="container"></div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        const container = document.getElementById('container');
        const info = document.getElementById('info');

        try {
            info.innerHTML = '<strong>間取り図 3Dビューア</strong><br>読込中...';

            // シーン・カメラ・レンダラー初期化
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(BACKGROUND_COLOR_PLACEHOLDER);

            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(15, 15, 15);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            // OrbitControls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            // ライト
            AMBIENT_LIGHT_PLACEHOLDER
            DIRECTIONAL_LIGHT_PLACEHOLDER

            // グリッド
            const gridHelper = new THREE.GridHelper(50, 50, 0x888888, 0xcccccc);
            scene.add(gridHelper);

            // JSON データ（埋め込み）
            const wallsData = JSON_DATA_PLACEHOLDER;
            
            let walls = wallsData.walls || [];
            
            // 3D表示用に壁データを2倍にスケーリング
            walls = walls.map(wall => ({
                ...wall,
                start: [wall.start[0] * 2, wall.start[1] * 2],
                end: [wall.end[0] * 2, wall.end[1] * 2],
                height: wall.height * 2,
                thickness: wall.thickness * 2,
                base_height: (wall.base_height || 0) * 2,
                windows: wall.windows ? wall.windows.map(w => ({
                    ...w,
                    position: w.position * 2,
                    width: w.width * 2,
                    height: w.height * 2
                })) : []
            }));
            
            // 照明データも2倍にスケーリング
            if (wallsData.metadata && wallsData.metadata.lights) {
                wallsData.metadata.lights = wallsData.metadata.lights.map(light => ({
                    ...light,
                    position: [
                        light.position[0] * 2,
                        light.position[1] * 2,
                        light.position[2] * 2
                    ]
                }));
            }
            
            // 図面の中心を計算してオフセット適用
            let offsetX = 0, offsetY = 0;
            if (walls.length > 0) {
                const allX = walls.flatMap(w => [w.start[0], w.end[0]]);
                const allY = walls.flatMap(w => [w.start[1], w.end[1]]);
                const minX = Math.min(...allX);
                const maxX = Math.max(...allX);
                const minY = Math.min(...allY);
                const maxY = Math.max(...allY);
                offsetX = (minX + maxX) / 2;
                offsetY = (minY + maxY) / 2;
            }
            
            info.innerHTML = `<strong>間取り図 3Dビューア</strong><br>壁数: ${walls.length}<br>マウス: 回転・拡大縮小・移動`;

            // 壁マテリアル
            const wallMaterial = new THREE.MeshStandardMaterial({
                color: 0xe0e0e0,
                roughness: 0.7,
                metalness: 0.1
            });

            walls.forEach(wall => {
                    const x1 = wall.start[0];
                    const y1 = wall.start[1];
                    const x2 = wall.end[0];
                    const y2 = wall.end[1];
                    const length = Math.sqrt((x2-x1)**2 + (y2-y1)**2);
                    const centerX = (x1 + x2) / 2;
                    const centerY = (y1 + y2) / 2;
                    
                    // base_heightを考慮したZ座標計算（窓対応）
                    const baseHeight = wall.base_height || 0;
                    const centerZ = baseHeight + (wall.height / 2);

                    // BoxGeometry (length, height, thickness)
                    const geometry = new THREE.BoxGeometry(length, wall.height, wall.thickness);
                    const mesh = new THREE.Mesh(geometry, wallMaterial);

                    // 位置（中心オフセットを適用して原点付近に配置、Y座標を反転して鏡像を修正）
                    mesh.position.set(centerX - offsetX, centerZ, -(centerY - offsetY));

                    // 回転（XZ平面上の角度、Y座標反転に合わせて調整）
                    const angle = Math.atan2(-(y2 - y1), x2 - x1);
                    mesh.rotation.y = angle;

                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    scene.add(mesh);
                });

                // 床生成（壁の範囲から計算）
                if (walls.length > 0) {
                    const allX = walls.flatMap(w => [w.start[0], w.end[0]]);
                    const allY = walls.flatMap(w => [w.start[1], w.end[1]]);
                    const minX = Math.min(...allX) - 1;
                    const maxX = Math.max(...allX) + 1;
                    const minY = Math.min(...allY) - 1;
                    const maxY = Math.max(...allY) + 1;

                    const floorW = maxX - minX;
                    const floorD = maxY - minY;
                    const floorGeometry = new THREE.BoxGeometry(floorW, 0.1, floorD);
                    const floorMaterial = new THREE.MeshStandardMaterial({ color: 0xd2b48c });
                    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
                    // 床もオフセットを適用してY座標を反転
                    floor.position.set((minX + maxX) / 2 - offsetX, -0.05, -((minY + maxY) / 2 - offsetY));
                    floor.receiveShadow = true;
                    scene.add(floor);
                    
                    WITH_LIGHTS_PLACEHOLDER
                }


            // アニメーションループ
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();

            // リサイズ対応
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });

        } catch (err) {
            info.innerHTML = `<strong>エラー発生</strong><br>${err.message}<br><small>コンソールを確認してください</small>`;
            console.error('3Dビューアエラー:', err);
        }
    </script>
</body>
</html>'''
    
    # 照明機能の有無によってコードを切り替え
    if with_lights:
        # 照明付きバージョン：暗い背景ライト、黒背景、ディレクショナルライトなし
        background_color = '0x000000'
        ambient_light_code = 'const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);\n            scene.add(ambientLight);'
        directional_light_code = '// ディレクショナルライトなし（スポットライトのみ）'
        
        lights_code = '''
                    // スポットライト配置
                    const lights = wallsData.metadata?.lights || [];
                    
                    // 天井生成
                    const avgWallHeight = walls.reduce((sum, w) => sum + w.height, 0) / walls.length || 2.7;
                    const ceilingGeometry = new THREE.BoxGeometry(floorW, 0.1, floorD);
                    const ceilingMaterial = new THREE.MeshStandardMaterial({ color: 0xf5f5f0 });
                    const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
                    ceiling.position.set((minX + maxX) / 2 - offsetX, avgWallHeight - 0.05, -((minY + maxY) / 2 - offsetY));
                    ceiling.receiveShadow = true;
                    scene.add(ceiling);
                    
                    // スポットライト配置
                    lights.forEach((light, idx) => {
                        // スポットライト本体
                        const spotLight = new THREE.SpotLight(0xffffff, 3.0);
                        spotLight.position.set(
                            light.position[0] - offsetX,
                            light.position[1],
                            -(light.position[2] - offsetY)
                        );
                        spotLight.target.position.set(
                            light.position[0] - offsetX,
                            0,
                            -(light.position[2] - offsetY)
                        );
                        spotLight.angle = Math.PI / 5;
                        spotLight.penumbra = 0.4;
                        spotLight.decay = 2;
                        spotLight.distance = 20;
                        spotLight.castShadow = true;
                        spotLight.shadow.mapSize.width = 1024;
                        spotLight.shadow.mapSize.height = 1024;
                        scene.add(spotLight);
                        scene.add(spotLight.target);
                        
                        // 照明位置を示す円柱オブジェクト（天井に設置）
                        const cylinderGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.3, 16);
                        const cylinderMaterial = new THREE.MeshStandardMaterial({ 
                            color: 0xffff00,
                            emissive: 0xffff00,
                            emissiveIntensity: 0.5
                        });
                        const cylinder = new THREE.Mesh(cylinderGeometry, cylinderMaterial);
                        // 天井の下面に設置（天井高さ - 0.15m）
                        cylinder.position.set(
                            light.position[0] - offsetX,
                            avgWallHeight - 0.15,
                            -(light.position[2] - offsetY)
                        );
                        scene.add(cylinder);
                    });
'''
    else:
        # 通常バージョン：明るい背景ライト、白背景、ディレクショナルライトあり
        background_color = '0xf0f0f0'
        ambient_light_code = 'const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);\n            scene.add(ambientLight);'
        directional_light_code = '''const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight.position.set(10, 20, 10);
            dirLight.castShadow = true;
            scene.add(dirLight);'''
        lights_code = '                    // 照明機能なし\n'
    
    # プレースホルダーを置換
    html_content = html_template.replace('JSON_DATA_PLACEHOLDER', json_content)
    html_content = html_content.replace('BACKGROUND_COLOR_PLACEHOLDER', background_color)
    html_content = html_content.replace('AMBIENT_LIGHT_PLACEHOLDER', ambient_light_code)
    html_content = html_content.replace('DIRECTIONAL_LIGHT_PLACEHOLDER', directional_light_code)
    html_content = html_content.replace('WITH_LIGHTS_PLACEHOLDER', lights_code)
    out_path.write_text(html_content, encoding='utf-8')
    return out_path


def _calc_distance(p1, p2):
    """2点間の距離を計算"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _calc_angle_diff(wall1, wall2):
    """2つの壁線の角度差を計算（度数法、180度回転も同一方向として扱う）"""
    dx1 = wall1['end'][0] - wall1['start'][0]
    dy1 = wall1['end'][1] - wall1['start'][1]
    angle1 = math.atan2(dy1, dx1) * 180 / math.pi
    
    dx2 = wall2['end'][0] - wall2['start'][0]
    dy2 = wall2['end'][1] - wall2['start'][1]
    angle2 = math.atan2(dy2, dx2) * 180 / math.pi
    
    # 角度差を計算
    diff = abs(angle1 - angle2)
    # 360度周期を考慮
    if diff > 180:
        diff = 360 - diff
    # 180度回転も同一方向として扱う（180度で剰余を取る）
    diff = min(diff, 180 - diff)
    return diff


def _determine_line_direction(p1, p2):
    """2点から線の方向を判定：縦(vertical)または横(horizontal)"""
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    return "horizontal" if dx > dy else "vertical"


def _add_line_to_json(json_data, p1, p2, wall_height=None, scale=50):
    """矩形選択から線を追加（2点から自動判定した方向で線を生成）"""
    import copy
    
    # 元データを保護するためディープコピー
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    # 既存の壁から平均厚さを取得
    thicknesses = [w.get('thickness', 0.12) for w in walls if 'thickness' in w]
    default_thickness = sum(thicknesses) / len(thicknesses) if thicknesses else 0.12
    
    # 既存の壁から平均高さを取得（指定がない場合）
    if wall_height is None:
        heights = [w.get('height', 2.4) for w in walls if 'height' in w]
        wall_height = sum(heights) / len(heights) if heights else 2.4
    
    # 矩形の座標を計算
    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
    
    # 可視化画像のメートル座標変換パラメータを取得
    all_x = [w['start'][0] for w in json_data['walls']] + [w['end'][0] for w in json_data['walls']]
    all_y = [w['start'][1] for w in json_data['walls']] + [w['end'][1] for w in json_data['walls']]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    margin = 50
    img_height = int((max_y - min_y) * scale) + 2 * margin
    
    # ピクセル座標 → メートル座標に変換
    x1_m = (x1 - margin) / scale + min_x
    y1_m = (img_height - y1 - margin) / scale + min_y
    x2_m = (x2 - margin) / scale + min_x
    y2_m = (img_height - y2 - margin) / scale + min_y
    
    # 方向を判定
    direction = _determine_line_direction(p1, p2)
    
    # 新しい壁線を生成
    if direction == "vertical":
        # 縦線：x座標を矩形の中央に固定、y座標は上下端
        x_center = (x1_m + x2_m) / 2
        start_pt = [x_center, min(y1_m, y2_m)]
        end_pt = [x_center, max(y1_m, y2_m)]
    else:  # horizontal
        # 横線：y座標を矩形の中央に固定、x座標は左右端
        y_center = (y1_m + y2_m) / 2
        start_pt = [min(x1_m, x2_m), y_center]
        end_pt = [max(x1_m, x2_m), y_center]
    
    # 線の長さを計算
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    length = round(math.sqrt(dx**2 + dy**2), 3)
    
    # 新しい壁のIDを生成（IDが文字列の場合も対応）
    try:
        max_id = max([int(w['id']) for w in walls], default=0)
    except (ValueError, TypeError):
        max_id = 0
    new_id = max_id + 1
    
    # 新しい壁オブジェクトを作成（既存の壁と同じ構造）
    new_wall = {
        'id': new_id,
        'start': [round(start_pt[0], 3), round(start_pt[1], 3)],
        'end': [round(end_pt[0], 3), round(end_pt[1], 3)],
        'height': round(wall_height, 3),  # 既存の壁と同じ高さ
        'base_height': 0.0,  # 通常の壁は床から
        'length': length,
        'thickness': round(default_thickness, 3),  # 既存の壁と同じ厚さ
        'source': 'added'  # 手動追加の壁として記録
    }
    
    # 壁を追加
    walls.append(new_wall)
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    
    return updated_data, direction, new_wall


def _point_in_rect(point, rect):
    """点が矩形内にあるかチェック（可視化画像のピクセル座標）"""
    x, y = point
    x_min, y_min = rect['left'], rect['top']
    x_max, y_max = rect['left'] + rect['width'], rect['top'] + rect['height']
    return x_min <= x <= x_max and y_min <= y <= y_max


def _line_intersects_rect(x1, y1, x2, y2, rect, tolerance=20):
    """線分が矩形と交差または近接しているかチェック（拡張版）"""
    x_min = rect['left'] - tolerance
    y_min = rect['top'] - tolerance
    x_max = rect['left'] + rect['width'] + tolerance
    y_max = rect['top'] + rect['height'] + tolerance
    
    # 1. 端点が矩形内にある
    if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or \
       (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
        return True
    
    # 2. 線分が矩形の辺と交差するか（簡易判定）
    # 線分が矩形を完全に横断している場合
    if (x1 < x_min and x2 > x_max) or (x2 < x_min and x1 > x_max) or \
       (y1 < y_min and y2 > y_max) or (y2 < y_min and y1 > y_max):
        return True
    
    # 3. 矩形が線分の間にある
    if (min(x1, x2) <= x_max and max(x1, x2) >= x_min) and \
       (min(y1, y2) <= y_max and max(y1, y2) >= y_min):
        return True
    
    return False


def _wall_in_rect(wall, rect, scale, margin, img_height, min_x, min_y, max_x, max_y):
    """壁線が矩形選択範囲内または近接しているかチェック（拡張版）"""
    # 壁線のメートル座標をピクセル座標に変換（visualize_3d_wallsと同じロジック）
    x1_px = int((wall['start'][0] - min_x) * scale) + margin
    y1_px = img_height - (int((wall['start'][1] - min_y) * scale) + margin)
    x2_px = int((wall['end'][0] - min_x) * scale) + margin
    y2_px = img_height - (int((wall['end'][1] - min_y) * scale) + margin)
    
    # 線分が矩形と交差または近接しているかチェック（許容範囲20ピクセル）
    return _line_intersects_rect(x1_px, y1_px, x2_px, y2_px, rect, tolerance=20)


def _filter_walls_strictly_in_rect(walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y):
    """
    矩形範囲内に完全に含まれる壁線のみを返す（精密フィルタリング）
    交差や近接ではなく、両端点が矩形内にある線のみを抽出
    """
    filtered_walls = []
    
    x_rect_min = rect['left']
    x_rect_max = rect['left'] + rect['width']
    y_rect_min = rect['top']
    y_rect_max = rect['top'] + rect['height']
    
    for wall in walls:
        # ピクセル座標に変換
        x1_px = int((wall['start'][0] - min_x) * scale) + margin
        y1_px = img_height - (int((wall['start'][1] - min_y) * scale) + margin)
        x2_px = int((wall['end'][0] - min_x) * scale) + margin
        y2_px = img_height - (int((wall['end'][1] - min_y) * scale) + margin)
        
        # 両端点が矩形内にあるかチェック（許容値なし）
        if (x_rect_min <= x1_px <= x_rect_max and y_rect_min <= y1_px <= y_rect_max and
            x_rect_min <= x2_px <= x_rect_max and y_rect_min <= y2_px <= y_rect_max):
            filtered_walls.append(wall)
    
    return filtered_walls


def _find_collinear_chains(walls_in_selection, distance_threshold=0.3, angle_threshold=15):
    """一直線上に並んだ連結壁線のチェーンを検出（3本以上の線を結合可能に）"""
    if len(walls_in_selection) < 2:
        return []
    
    # 各壁線間の接続情報を構築
    connections = {}  # {wall_id: [(connected_wall_id, connection_type, distance), ...]}
    
    for i, wall1 in enumerate(walls_in_selection):
        wall1_id = wall1['id']
        if wall1_id not in connections:
            connections[wall1_id] = []
        
        for j, wall2 in enumerate(walls_in_selection):
            if i >= j:
                continue
            
            wall2_id = wall2['id']
            if wall2_id not in connections:
                connections[wall2_id] = []
            
            # 角度が同一直線上かチェック
            angle_diff = _calc_angle_diff(wall1, wall2)
            if angle_diff >= angle_threshold:
                continue
            
            # 4つの端点組み合わせをチェック
            endpoint_pairs = [
                (wall1['end'], wall2['start'], 'end-start'),
                (wall1['end'], wall2['end'], 'end-end'),
                (wall1['start'], wall2['start'], 'start-start'),
                (wall1['start'], wall2['end'], 'start-end'),
            ]
            
            for p1, p2, connection_type in endpoint_pairs:
                distance = _calc_distance(p1, p2)
                if distance < distance_threshold:
                    connections[wall1_id].append((wall2_id, connection_type, distance))
                    # 逆方向の接続も記録
                    reverse_type = connection_type.split('-')[::-1]
                    reverse_type = f"{reverse_type[0]}-{reverse_type[1]}"
                    connections[wall2_id].append((wall1_id, reverse_type, distance))
                    break
    
    # 連結チェーンを検出（DFS）
    visited = set()
    chains = []
    
    def build_chain(start_wall_id, current_chain, visited_in_chain):
        """再帰的にチェーンを構築"""
        if start_wall_id in visited_in_chain:
            return
        
        visited_in_chain.add(start_wall_id)
        current_chain.append(start_wall_id)
        
        # 接続された壁線を探索
        if start_wall_id in connections:
            for connected_id, conn_type, dist in connections[start_wall_id]:
                if connected_id not in visited_in_chain:
                    build_chain(connected_id, current_chain, visited_in_chain)
    
    # 各壁線からチェーンを構築
    for wall in walls_in_selection:
        wall_id = wall['id']
        if wall_id not in visited:
            chain = []
            visited_in_chain = set()
            build_chain(wall_id, chain, visited_in_chain)
            
            if len(chain) >= 2:  # 2本以上のチェーン
                visited.update(chain)
                # チェーン内の壁線オブジェクトを取得
                chain_walls = [w for w in walls_in_selection if w['id'] in chain]
                chains.append(chain_walls)
    
    return chains


def _find_mergeable_walls(walls_in_selection, distance_threshold=0.3, angle_threshold=15):
    """選択範囲内で結合可能な壁線ペアまたはチェーンを探す（3本以上も対応）"""
    candidates = []
    
    # まず、一直線上の連結チェーンを検出（3本以上対応）
    chains = _find_collinear_chains(walls_in_selection, distance_threshold, angle_threshold)
    
    # チェーンを結合候補として追加
    for chain in chains:
        if len(chain) >= 2:
            # チェーンの最初と最後の壁線を取得
            first_wall = chain[0]
            last_wall = chain[-1]
            
            # チェーン全体の端点を決定
            # 最初の壁線と最後の壁線の端点から、最も離れた2点を選ぶ
            all_endpoints = [
                first_wall['start'],
                first_wall['end'],
                last_wall['start'],
                last_wall['end']
            ]
            
            max_dist = 0
            chain_start = None
            chain_end = None
            
            for i, p1 in enumerate(all_endpoints):
                for j, p2 in enumerate(all_endpoints):
                    if i >= j:
                        continue
                    dist = _calc_distance(p1, p2)
                    if dist > max_dist:
                        max_dist = dist
                        chain_start = p1
                        chain_end = p2
            
            # チェーン全体の平均角度差を計算
            total_angle_diff = 0
            for i in range(len(chain) - 1):
                total_angle_diff += _calc_angle_diff(chain[i], chain[i+1])
            avg_angle_diff = total_angle_diff / (len(chain) - 1) if len(chain) > 1 else 0
            
            candidates.append({
                'walls': chain,  # チェーン全体
                'is_chain': True,
                'chain_length': len(chain),
                'distance': max_dist,
                'angle_diff': avg_angle_diff,
                'new_start': chain_start,
                'new_end': chain_end,
                'confidence': 1.0
            })
    
    # 個別のペアも検出（従来の機能を維持）
    for i, wall1 in enumerate(walls_in_selection):
        for j, wall2 in enumerate(walls_in_selection):
            if i >= j:
                continue
            
            # 4つの端点組み合わせをチェック
            connections = [
                (wall1['end'], wall2['start'], 'end-start', wall1['end'], wall2['end']),
                (wall1['end'], wall2['end'], 'end-end', wall1['end'], wall2['start']),
                (wall1['start'], wall2['start'], 'start-start', wall1['end'], wall2['end']),
                (wall1['start'], wall2['end'], 'start-end', wall1['end'], wall2['start']),
            ]
            
            for p1, p2, connection_type, new_p1, new_p2 in connections:
                distance = _calc_distance(p1, p2)
                angle_diff = _calc_angle_diff(wall1, wall2)
                
                # 距離と角度の両方でチェック（180度回転も同一方向として扱う）
                if distance < distance_threshold and angle_diff < angle_threshold:
                    candidates.append({
                        'wall1': wall1,
                        'wall2': wall2,
                        'is_chain': False,
                        'distance': distance,
                        'angle_diff': angle_diff,
                        'connection': connection_type,
                        'new_start': new_p1,
                        'new_end': new_p2,
                        'confidence': 1.0 - (distance / distance_threshold)
                    })
    
    # 信頼度でソート
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    return candidates


def _merge_walls_in_json(json_data, merge_pairs):
    """JSONデータ内の壁線を結合（ディープコピーで元データを保護、3本以上のチェーンにも対応）"""
    import copy
    # 元のデータを変更しないようにディープコピーを作成
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    for pair in merge_pairs:
        # チェーン結合（3本以上）の場合
        if pair.get('is_chain', False) and 'walls' in pair:
            chain_walls = pair['walls']
            if len(chain_walls) < 2:
                continue
            
            # 最初の壁線を残して更新、他は削除
            first_wall_id = chain_walls[0]['id']
            other_wall_ids = [w['id'] for w in chain_walls[1:]]
            
            # 最初の壁線を探して座標を更新
            for wall in walls:
                if wall['id'] == first_wall_id:
                    wall['start'] = pair['new_start']
                    wall['end'] = pair['new_end']
                    
                    # 長さを再計算
                    dx = wall['end'][0] - wall['start'][0]
                    dy = wall['end'][1] - wall['start'][1]
                    wall['length'] = round(math.sqrt(dx**2 + dy**2), 3)
                    break
            
            # チェーン内の他の壁線を削除
            walls[:] = [w for w in walls if w['id'] not in other_wall_ids]
        
        # 通常のペア結合（2本）の場合
        elif 'wall1' in pair and 'wall2' in pair:
            wall1_id = pair['wall1']['id']
            wall2_id = pair['wall2']['id']
            
            # wall1を探して更新
            for wall in walls:
                if wall['id'] == wall1_id:
                    # 結合タイプに応じて座標を更新
                    if pair['connection'] == 'end-start':
                        wall['end'] = pair['wall2']['end']
                    elif pair['connection'] == 'end-end':
                        wall['end'] = pair['wall2']['start']
                    elif pair['connection'] == 'start-start':
                        wall['start'] = pair['wall2']['end']
                    elif pair['connection'] == 'start-end':
                        wall['start'] = pair['wall2']['start']
                    
                    # 長さを再計算
                    dx = wall['end'][0] - wall['start'][0]
                    dy = wall['end'][1] - wall['start'][1]
                    wall['length'] = round(math.sqrt(dx**2 + dy**2), 3)
                    break
            
            # wall2を削除
            walls[:] = [w for w in walls if w['id'] != wall2_id]
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    return updated_data


def _delete_walls_in_json(json_data, wall_ids_to_delete):
    """JSONデータ内の指定された壁線を削除"""
    import copy
    # 元のデータを変更しないようにディープコピーを作成
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    # 削除対象の壁IDセット化（高速な検索のため）
    delete_ids = set(wall_ids_to_delete)
    
    # 指定された壁を削除
    walls[:] = [w for w in walls if w['id'] not in delete_ids]
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    return updated_data


def _add_window_walls(json_data, wall1, wall2, window_height, base_height, room_height):
    """
    窓で分断された2本の壁の間に、床側と天井側の壁を追加
    
    Args:
        json_data: 元のJSONデータ
        wall1, wall2: 窓で分断された2本の壁
        window_height: 窓の高さ（m）
        base_height: 床から窓下端までの高さ（m）
        room_height: 部屋の天井高さ（m）
    
    Returns:
        更新されたJSONデータ、追加された壁のリスト
    """
    import copy
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    # 2本の壁の端点から窓の位置を計算
    # wall1の端点とwall2の端点で最も近い組み合わせを見つける
    endpoints = [
        (wall1['start'], wall2['start']),
        (wall1['start'], wall2['end']),
        (wall1['end'], wall2['start']),
        (wall1['end'], wall2['end']),
    ]
    
    min_dist = float('inf')
    window_start = None
    window_end = None
    
    for p1, p2 in endpoints:
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if dist < min_dist:
            min_dist = dist
            window_start = p1
            window_end = p2
    
    # 既存の壁から平均厚さを取得
    thicknesses = [w.get('thickness', 0.12) for w in walls if 'thickness' in w]
    default_thickness = sum(thicknesses) / len(thicknesses) if thicknesses else 0.12
    
    # 新しい壁のIDを生成
    try:
        max_id = max([int(w['id']) for w in walls], default=0)
    except (ValueError, TypeError):
        max_id = 0
    
    added_walls = []
    
    # 床側の壁を追加（床〜窓下端）
    floor_wall = {
        'id': max_id + 1,
        'start': [round(window_start[0], 3), round(window_start[1], 3)],
        'end': [round(window_end[0], 3), round(window_end[1], 3)],
        'height': round(base_height, 3),
        'base_height': 0.0,  # 床から
        'length': round(min_dist, 3),
        'thickness': round(default_thickness, 3),
        'source': 'window_added'
    }
    walls.append(floor_wall)
    added_walls.append(floor_wall)
    
    # 天井側の壁を追加（窓上端〜天井）
    ceiling_height = room_height - (base_height + window_height)
    ceiling_wall = {
        'id': max_id + 2,
        'start': [round(window_start[0], 3), round(window_start[1], 3)],
        'end': [round(window_end[0], 3), round(window_end[1], 3)],
        'height': round(ceiling_height, 3),
        'base_height': round(base_height + window_height, 3),  # 窓上端から
        'length': round(min_dist, 3),
        'thickness': round(default_thickness, 3),
        'source': 'window_added'
    }
    walls.append(ceiling_wall)
    added_walls.append(ceiling_wall)
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    
    return updated_data, added_walls


def _find_closest_wall_to_point(walls, point_px, scale, margin, img_height, min_x, min_y, max_x, max_y):
    """ポイントから最も近い壁を見つける"""
    min_distance = float('inf')
    closest_wall = None
    
    # ポイントをメートル座標に変換
    point_m = [
        (point_px[0] - margin) / scale + min_x,
        (img_height - point_px[1] - margin) / scale + min_y
    ]
    
    for wall in walls:
        # 壁の両端点を取得（安全にアンパック）
        try:
            start = wall.get('start')
            end = wall.get('end')
            
            # 配列形式であることを確認
            if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)):
                continue
            if len(start) < 2 or len(end) < 2:
                continue
            
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
        except (TypeError, ValueError, KeyError):
            continue
        
        # ポイントから線分までの最短距離を計算
        # 線分上の最近点を見つける
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # 点と点の距離
            distance = math.sqrt((point_m[0] - x1)**2 + (point_m[1] - y1)**2)
        else:
            # 線分上での最近点を計算
            t = max(0, min(1, ((point_m[0] - x1) * dx + (point_m[1] - y1) * dy) / (dx**2 + dy**2)))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            distance = math.sqrt((point_m[0] - closest_x)**2 + (point_m[1] - closest_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_wall = wall
    
    return closest_wall, min_distance


def _generate_blender_script(json_path: Path, out_path: Path) -> Path:
    """テンプレート(blender_import_walls.py)からJSONパスを書き換えて出力"""
    tpl_path = BASE_DIR / "blender_import_walls.py"
    if not tpl_path.exists():
        raise FileNotFoundError("blender_import_walls.py が見つかりません")

    content = tpl_path.read_text(encoding="utf-8")

    # 既存の json_path = r"..." の代入行を置換（関数replで安全に挿入）
    pattern = r"(?m)^json_path\s*=\s*r\".*\"\s*$"
    replacement_line = f'json_path = r"{str(json_path)}"'

    try:
        if re.search(pattern, content):
            # 関数replを使うことで、置換文字列中のバックスラッシュが解釈されない
            new_content = re.sub(pattern, lambda m: replacement_line, content)
        else:
            # フォールバック: 末尾に安全に追記
            appended = (
                "\n\n# 追加: 自動生成により設定\n"
                f"json_path = r\"{str(json_path)}\"\n"
                "if Path(json_path).exists():\n"
                "    import_walls_from_json(json_path, create_floor=True)\n"
                "else:\n"
                "    print(\"Error: File not found: {}\".format(json_path))\n"
            )
            new_content = content + appended
    except re.error:
        # 正規表現トラブル時は行ベース置換にフォールバック
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("json_path"):
                lines[i] = replacement_line
                break
        else:
            lines.append("")
            lines.append("# 追加: 自動生成により設定")
            lines.append(replacement_line)
            lines.append("if Path(json_path).exists():")
            lines.append("    import_walls_from_json(json_path, create_floor=True)")
            lines.append("else:")
            lines.append("    print(\"Error: File not found: {}\".format(json_path))")
        new_content = "\n".join(lines)

    out_path.write_text(new_content, encoding="utf-8")
    return out_path


def main():
    st.set_page_config(page_title="図面→Blender 変換ツール", layout="wide")
    st.title("図面(PDF/JPG/PNG) → Blenderスクリプト 生成ツール")
    st.caption("PDF/画像をアップロードして、3Dデータ(JSON)とBlenderスクリプトを作成します。")
    
    # 固定画像幅（自動結合と手動編集で統一）
    DISPLAY_IMAGE_WIDTH = 800

    # セッション状態の初期化（結果の永続化）
    if "processed" not in st.session_state:
        st.session_state.processed = False
    for key in [
        "out_dir", "refined_img", "refined_name", "refined_bytes", "json_bytes", "json_name",
        "blender_bytes", "blender_name", "viz_bytes", "viz_name", "viewer_html_bytes", "viewer_html_name",
        "zip_bytes", "zip_name", "merged_json_bytes", "merged_json_name", "merged_viz_bytes", "merged_viz_name",
        "merged_processed"
    ]:
        st.session_state.setdefault(key, None)
    if "merged_processed" not in st.session_state:
        st.session_state.merged_processed = False
    if "viz_scale" not in st.session_state:
        st.session_state.viz_scale = 50  # 可視化スケール（メートル→ピクセル）

    with st.sidebar:
        st.header("パラメータ設定")
        
        st.subheader("📋 入力ファイル形式に応じたスケール")
        file_info = st.selectbox(
            "ファイル形式を選択して説明を表示",
            ["① PDF", "② JPG/PNG"],
            help="ファイル形式によってスケール計算が異なります"
        )
        
        if file_info == "① PDF":
            st.info(
                "• DPIスライダーを高くするとPDF全体が拡大され、ピクセル数が増えます\n\n"
                "• 300dpi → 高い精度で壁を検出\n"
                "• 150dpi → 高速処理、ノイズ増加"
            )
        else:
            st.info(
                "• 画像はそのまま利用されます（DPIスライダーは無視）\n\n"
                "• 入力画像の解像度がそのまま「ピクセル → メートル」の計算に使われます"
            )
        
        st.divider()
        dpi = st.slider(
            "📐 PDFレンダリングDPI",
            min_value=150,
            max_value=600,
            value=300,
            step=50,
            help="PDF→画像変換時の解像度。高いほど詳細だが処理が遅くなります。JPG/PNG使用時は無視されます。"
        )
        
        st.subheader("🎨 壁検出パラメータ")
        
        st.markdown("#### 黒閾値（0〜255）")
        with st.expander("💡 調整のコツ", expanded=False):
            st.markdown(
                "**調整方法:**\n"
                "1. 最初は190から試す\n"
                "2. ノイズが多い → 値を下げる\n"
                "3. 薄い線が消える → 値を上げる"
            )
        
        black_threshold = st.slider(
            "黒閾値",
            min_value=100,
            max_value=240,
            value=190,
            step=5,
            help="値を上げる = 薄い線も拾う。値を下げる = 濃い線のみ。"
        )
        
        st.markdown("#### 最小線幅（ピクセル）")
        with st.expander("💡 調整のコツ", expanded=False):
            st.markdown(
                "**調整方法:**\n"
                "1. 壁が多すぎる → 値を上げる\n"
                "2. 壁が足りない → 値を下げる"
            )
        
        min_thickness = st.slider(
            "最小線幅(px)",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            help="小さい = 細い線も拾う。大きい = 太い線のみ。DPI高い時は大きく。"
        )
        
        # ピクセル→メートル係数を固定値に設定
        pixel_to_meter = 0.005
        
        st.subheader("🏗️ Blender出力スケール")
        wall_height = st.number_input(
            "壁高さ(部屋の天井高さ)",
            min_value=0.1,
            max_value=10.0,
            value=2.4,
            step=0.1,
            help="部屋の天井高さ（床から天井までの高さ）をメートル単位で指定します。一般的な住宅は2.4m程度です。"
        )
        
        st.subheader("👁️ 可視化設定")
        viz_scale = st.slider(
            "2D可視化スケール(px/m)",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
            help="可視化PNG上で1メートルが何ピクセルで表示されるか。大きいほど拡大表示。"
        )
        # セッション状態に保存（手動編集で使用）
        st.session_state.viz_scale = viz_scale
        
        st.divider()
        st.subheader("💡 スケール調整のコツ")
        st.markdown(
            "**壁のサイズが正しくない場合:**\n\n"
            "1. **PDF 高解像度（300dpi）** → pixel_to_meter = 0.005 〜 0.008\n"
            "2. **PDF 低解像度（150dpi）** → pixel_to_meter = 0.015 〜 0.02\n"
            "3. **JPG/PNG スキャン品質** → 原本の解像度を確認して調整\n\n"
            "詳しくは README.md の **トラブルシューティング** を参照してください。"
        )

    uploaded = st.file_uploader("図面PDF/画像をアップロード", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=False)
    
    # PDFの場合、ページ数を確認してページ選択UIを表示
    page_number = 0  # デフォルト
    if uploaded is not None and uploaded.name.lower().endswith('.pdf'):
        try:
            # 一時的にPDFを保存してページ数を取得
            temp_pdf = io.BytesIO(uploaded.getvalue())
            doc = fitz.open(stream=temp_pdf, filetype="pdf")
            total_pages = len(doc)
            doc.close()
            
            if total_pages > 1:
                st.info(f"📄 このPDFは {total_pages} ページあります。処理するページを選択してください。")
                page_number = st.selectbox(
                    "処理するページを選択",
                    options=list(range(total_pages)),
                    format_func=lambda x: f"ページ {x + 1}",
                    help="PDFの何ページ目を処理するか選択します（0始まり）"
                )
            else:
                st.info("📄 このPDFは 1 ページです。")
        except Exception as e:
            st.warning(f"PDFページ数の取得に失敗しました: {e}")
    
    if uploaded is None and not st.session_state.processed:
        st.info("PDF/JPG/PNGをアップロードしてください。")

    # 実行ボタン
    run = st.button("変換を実行", type="primary")
    # run=Falseでも過去結果は表示するため、早期returnはしない

    # 途中経過の表示可否（Falseで最終結果のみ表示）
    show_progress = False

    if run:
        if uploaded is None:
            st.error("ファイルが未選択です。先にPDF/画像をアップロードしてください。")
            return
        # 出力ディレクトリ（タイムスタンプで分離）
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUTS_DIR / f"run_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 入力ファイル保存
        input_suffix = Path(uploaded.name).suffix.lower()
        input_path = out_dir / f"input{input_suffix}"
        _save_uploaded_file(uploaded, input_path)
        if show_progress:
            st.success(f"入力を保存: {input_path}")

        # PDFなら画像へレンダリング
        if input_suffix == ".pdf":
            if show_progress:
                st.write(f"PDFを画像に変換中… (ページ {page_number + 1})")
            source_image = out_dir / "source.png"
            pdf_to_image(str(input_path), output_path=str(source_image), page_number=page_number, dpi=dpi)
            image_path = source_image
        else:
            image_path = input_path

        if show_progress:
            st.write("壁線抽出とノイズ除去を実行中…")
        try:
            refined_img, refined_path = refine_floor_plan_from_image(
                str(image_path), black_threshold=black_threshold, min_thickness=min_thickness, remove_corners=False
            )
            refined_path = Path(refined_path)
            
            # refined画像をout_dirに移動
            refined_dest = out_dir / refined_path.name
            if refined_path.exists() and refined_path != refined_dest:
                refined_path.rename(refined_dest)
                refined_path = refined_dest
                if show_progress:
                    st.success(f"壁線抽出完了: {refined_path.name}")
        except Exception as e:
            st.error(f"壁線抽出でエラー: {e}")
            return

        # 3D JSON 生成
        if show_progress:
            st.write("3D座標JSONを生成中…")
        json_path = out_dir / "walls_3d.json"
        try:
            result = process_image_to_3d(str(refined_path), str(json_path), wall_height=wall_height, pixel_to_meter=pixel_to_meter)
            if result is None:
                st.error("3D座標の生成に失敗しました。")
                return
        except Exception as e:
            st.error(f"3D座標生成でエラー: {e}")
            return

        # 2D可視化
        if show_progress:
            st.write("2D可視化画像を生成中…")
        viz_path = out_dir / "visualization.png"
        try:
            canvas = visualize_3d_walls(str(json_path), str(viz_path), scale=int(viz_scale))
        except Exception as e:
            canvas = None
            st.warning(f"可視化の生成に失敗しました: {e}")

        # Blenderスクリプト生成
        if show_progress:
            st.write("Blenderインポートスクリプトを生成中…")
        blender_script = out_dir / "blender_import_autogen.py"
        try:
            _generate_blender_script(json_path, blender_script)
        except Exception as e:
            st.error(f"Blenderスクリプト生成でエラー: {e}")
            return

        # Three.js HTMLビューア生成
        if show_progress:
            st.write("3DビューアHTML(Three.js)を生成中…")
        viewer_html = out_dir / "viewer_3d.html"
        try:
            _generate_3d_viewer_html(json_path, viewer_html)
        except Exception as e:
            st.error(f"3DビューアHTML生成でエラー: {e}")
            return

        # セッションに結果を保存（ダウンロードで再実行されても残す）
        st.session_state.out_dir = str(out_dir)
        st.session_state.refined_img = refined_img
        st.session_state.refined_name = refined_path.name
        st.session_state.refined_bytes = refined_path.read_bytes()
        st.session_state.json_bytes = json_path.read_bytes()
        st.session_state.json_name = json_path.name
        st.session_state.blender_bytes = blender_script.read_bytes()
        st.session_state.blender_name = blender_script.name
        st.session_state.viewer_html_bytes = viewer_html.read_bytes()
        st.session_state.viewer_html_name = viewer_html.name
        if canvas is not None and viz_path.exists():
            st.session_state.viz_bytes = viz_path.read_bytes()
            st.session_state.viz_name = viz_path.name
        else:
            st.session_state.viz_bytes = None
            st.session_state.viz_name = None
        # ZIP生成（JSON+Blenderスクリプト+抽出PNG+3DビューアHTML）
        try:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(st.session_state.json_name, st.session_state.json_bytes)
                zf.writestr(st.session_state.blender_name, st.session_state.blender_bytes)
                zf.writestr(st.session_state.viewer_html_name, st.session_state.viewer_html_bytes)
                if st.session_state.refined_bytes and st.session_state.refined_name:
                    zf.writestr(st.session_state.refined_name, st.session_state.refined_bytes)
            zip_buf.seek(0)
            st.session_state.zip_bytes = zip_buf.getvalue()
            st.session_state.zip_name = f"{out_dir.name}_bundle.zip"
        except Exception:
            st.session_state.zip_bytes = None
            st.session_state.zip_name = None
        st.session_state.processed = True
        st.success("変換が完了しました！")

    # セッションに結果があれば常に表示（ダウンロードでの再実行でも消えない）
    if st.session_state.processed:
        # 画像表示
        if st.session_state.refined_img is not None:
            st.image(st.session_state.refined_img, caption=f"壁線抽出結果: {st.session_state.refined_name}", clamp=True)
        # ダウンロードボタン
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.session_state.json_bytes:
                st.download_button(
                    label="JSON",
                    data=st.session_state.json_bytes,
                    file_name=st.session_state.json_name,
                    mime="application/json",
                )
        with col2:
            if st.session_state.blender_bytes:
                st.download_button(
                    label="Blenderスクリプト",
                    data=st.session_state.blender_bytes,
                    file_name=st.session_state.blender_name,
                    mime="text/x-python",
                )
        with col3:
            if st.session_state.viewer_html_bytes:
                st.download_button(
                    label="3DビューアHTML★",
                    data=st.session_state.viewer_html_bytes,
                    file_name=st.session_state.viewer_html_name,
                    mime="text/html",
                )
            # 照明付き3Dビューア
            if st.session_state.get('viewer_html_lights_bytes'):
                st.download_button(
                    label="💡照明付き3DHTML",
                    data=st.session_state.viewer_html_lights_bytes,
                    file_name=st.session_state.viewer_html_lights_name,
                    mime="text/html",
                )
        with col4:
            if st.session_state.refined_bytes:
                st.download_button(
                    label="壁線抽出PNG",
                    data=st.session_state.refined_bytes,
                    file_name=st.session_state.refined_name,
                    mime="image/png",
                )
        with col5:
            if st.session_state.viz_bytes:
                st.download_button(
                    label="可視化PNG",
                    data=st.session_state.viz_bytes,
                    file_name=st.session_state.viz_name,
                    mime="image/png",
                )
        # ZIP一括ダウンロード
        if st.session_state.zip_bytes:
            st.download_button(
                label="📦 全ファイル一括ダウンロード（ZIP: JSON+Blenderスクリプト+3DビューアHTML+抽出PNG）",
                data=st.session_state.zip_bytes,
                file_name=st.session_state.zip_name,
                mime="application/zip",
            )

        # 自動壁結合セクション
        st.divider()
        st.subheader("🤖 自動壁結合（refined画像を基準）")
        st.write("refined画像の骨格線を基準に、分裂した壁を自動的に結合します。")
        
        col_merge1, col_merge2 = st.columns(2)
        with col_merge1:
            merge_radius = st.slider("結合検索範囲（ピクセル）", min_value=30, max_value=150, value=50, step=10)
        with col_merge2:
            merge_angle = st.slider("角度許容範囲（度）", min_value=5, max_value=45, value=15, step=5)
        
        if st.button("🔗 自動結合を実行", type="secondary"):
            if st.session_state.json_bytes is None or st.session_state.out_dir is None:
                st.error("先に「変換を実行」を完了してください。")
            else:
                try:
                    out_dir = Path(st.session_state.out_dir)
                    json_path = out_dir / st.session_state.json_name
                    refined_path = out_dir / st.session_state.refined_name
                    merged_json_path = out_dir / "walls_3d_merged.json"
                    merged_viz_path = out_dir / "visualization_merged.png"
                    
                    with st.spinner("自動結合を実行中…"):
                        # 自動結合実行
                        merger = WallAutoMerger(search_radius=merge_radius, angle_tolerance=merge_angle)
                        merger.process(str(refined_path), str(json_path), str(merged_json_path))
                        
                        # 結合後の可視化を生成
                        # 可視化スケールはユーザ設定値を使用（結合前後で画像サイズを揃える）
                        visualize_3d_walls(
                            str(merged_json_path),
                            str(merged_viz_path),
                            scale=int(st.session_state.viz_scale)
                        )
                    
                    # 結合結果をセッションに保存
                    st.session_state.merged_json_bytes = merged_json_path.read_bytes()
                    st.session_state.merged_json_name = merged_json_path.name
                    if merged_viz_path.exists():
                        st.session_state.merged_viz_bytes = merged_viz_path.read_bytes()
                        st.session_state.merged_viz_name = merged_viz_path.name
                    st.session_state.merged_processed = True
                    st.success("自動結合が完了しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"自動結合でエラー: {e}")
        
        # 結合結果の表示
        if st.session_state.merged_processed:
            st.divider()
            st.subheader("📊 結合結果")
            
            # 結合前後の統計を比較
            col_before, col_after = st.columns(2)
            with col_before:
                st.write("**結合前**")
                try:
                    info_before = json.loads(st.session_state.json_bytes.decode("utf-8"))
                    st.metric("壁セグメント数", info_before.get("metadata", {}).get("total_walls"))
                except Exception:
                    pass
            
            with col_after:
                st.write("**結合後**")
                try:
                    info_after = json.loads(st.session_state.merged_json_bytes.decode("utf-8"))
                    walls_after = info_after.get("metadata", {}).get("total_walls")
                    st.metric("壁セグメント数", walls_after)
                except Exception:
                    pass
            
            # 可視化比較表示
            col_viz_before, col_viz_after = st.columns(2)
            with col_viz_before:
                st.write("**結合前の可視化**")
                if st.session_state.viz_bytes:
                    st.image(st.session_state.viz_bytes, caption="結合前", width=DISPLAY_IMAGE_WIDTH)
            
            with col_viz_after:
                st.write("**結合後の可視化**")
                if st.session_state.merged_viz_bytes:
                    st.image(st.session_state.merged_viz_bytes, caption="結合後", width=DISPLAY_IMAGE_WIDTH)
            
            # 座標比較セクション（デバッグ用）
            with st.expander("🔍 座標比較（デバッグ）"):
                st.write("**自動結合前後の壁座標を比較：**")
                try:
                    info_before = json.loads(st.session_state.json_bytes.decode("utf-8"))
                    info_after = json.loads(st.session_state.merged_json_bytes.decode("utf-8"))
                    
                    walls_before = info_before.get("walls", [])
                    walls_after = info_after.get("walls", [])
                    
                    # 最初の3つの壁を比較表示
                    num_to_compare = min(3, len(walls_before))
                    
                    for i in range(num_to_compare):
                        if i < len(walls_before):
                            wall_b = walls_before[i]
                            st.write(f"**Wall {i} (結合前):**")
                            col_id, col_start, col_end = st.columns(3)
                            with col_id:
                                st.text(f"ID: {wall_b.get('id')}")
                            with col_start:
                                st.text(f"Start: {wall_b.get('start')}")
                            with col_end:
                                st.text(f"End: {wall_b.get('end')}")
                    
                    st.write("---")
                    
                    for i in range(num_to_compare):
                        if i < len(walls_after):
                            wall_a = walls_after[i]
                            st.write(f"**Wall {i} (結合後):**")
                            col_id, col_start, col_end = st.columns(3)
                            with col_id:
                                st.text(f"ID: {wall_a.get('id')}")
                            with col_start:
                                st.text(f"Start: {wall_a.get('start')}")
                            with col_end:
                                st.text(f"End: {wall_a.get('end')}")
                    
                    st.info(f"※ 結合前: {len(walls_before)}壁 → 結合後: {len(walls_after)}壁")
                except Exception as e:
                    st.error(f"座標比較でエラー: {e}")
            
            # 結合後ファイルのダウンロード
            st.write("**結合後のファイルをダウンロード**")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if st.session_state.merged_json_bytes:
                    st.download_button(
                        label="結合後JSON",
                        data=st.session_state.merged_json_bytes,
                        file_name=st.session_state.merged_json_name,
                        mime="application/json",
                    )
            with col_dl2:
                if st.session_state.merged_viz_bytes:
                    st.download_button(
                        label="結合後の可視化PNG",
                        data=st.session_state.merged_viz_bytes,
                        file_name=st.session_state.merged_viz_name,
                        mime="image/png",
                    )
            
            # 保存/キャンセルボタン
            st.write("**結合を保存しますか？**")
            col_save_btn1, col_save_btn2 = st.columns(2)
            with col_save_btn1:
                if st.button("✅ 保存", type="primary", key="save_merge"):
                    # 結合後のJSONをセッション状態に保存
                    st.session_state.json_bytes = st.session_state.merged_json_bytes
                    st.session_state.json_name = st.session_state.merged_json_name
                    st.session_state.viz_bytes = st.session_state.merged_viz_bytes
                    st.session_state.viz_name = st.session_state.merged_viz_name
                    st.session_state.merged_processed = False  # 結合セクションを非表示
                    st.success("結合結果を保存しました。手動編集セクションで編集できます。")
                    st.rerun()
            
            with col_save_btn2:
                if st.button("❌ キャンセル", key="cancel_merge"):
                    st.session_state.merged_processed = False
                    st.info("結合をキャンセルしました。")
                    st.rerun()
        
        # 壁線手動編集モード
        st.divider()
        st.subheader("🔧 壁線手動編集")
        
        # モード選択タブ
        edit_mode = st.radio(
            "編集モードを選択:",
            ["線を結合", "窓を追加して結合", "線を追加", "線を削除", "スケール校正", "照明を配置"],
            horizontal=True,
            help="線を結合：2つの壁線を繋ぐ\n窓を追加して結合：窓で分断された2本の壁を上下の壁で繋ぐ\n線を追加：新しい壁線を追加\n線を削除：選択範囲の壁を削除\nスケール校正：線の長さから実寸を入力してサイズを調整\n照明を配置：クリック位置にスポットライトを配置"
        )
        
        if edit_mode == "線を結合":
            st.markdown(
                "可視化画像上で**クリックして結合範囲を指定**すると、自動的に近接する壁線を結合します。"
                "**一直線上に並んだ3本以上の線も一度に結合できます。**"
            )
        elif edit_mode == "窓を追加して結合":
            st.markdown(
                "可視化画像上で**窓で分断された2本の壁を矩形で囲む**と、その間に床側と天井側の壁を追加します。\n"
                "**窓のサイズ（高さと床からの距離）を入力**して、開口部の上下に壁を生成します。"
            )
        elif edit_mode == "線を追加":
            st.markdown(
                "可視化画像上で**クリックして追加範囲を指定**すると、矩形の方向に応じた線が自動生成されます。\n"
                "**縦矩形**→縦線、**横矩形**→横線が矩形の中央に追加されます。"
            )
        elif edit_mode == "線を削除":
            st.markdown(
                "可視化画像上で**クリックして削除対象を指定**すると、矩形で囲んだ範囲に完全に含まれる**全ての壁線が削除**されます。"
            )
        else:  # スケール校正
            st.markdown(
                "可視化画像上で**1本の壁線を矩形で囲む**と、その線のピクセル長を自動測定。\n"
                "**その線が実際に何マス（90cm単位）に相当するかを入力**すると、スケール値を自動計算して再生成します。"
            )
        
        with st.expander("💡 使い方", expanded=False):
            if edit_mode == "線を結合":
                st.markdown(
                    "**複数線結合の手順:**\n"
                    "1. 下の画像上で**2回クリック**して矩形の対角を指定\n"
                    "2. 結合したい壁線を**矩形で囲む**（2本でも3本以上でも可）\n"
                    "3. 「➕ この選択を追加」ボタンで追加（色が変わります）\n"
                    "4. さらに結合したい箇所があれば手順1-3を繰り返す\n"
                    "5. 「🔗 結合実行」で全ての選択範囲を一括結合\n\n"
                    "**特徴:**\n"
                    "- **一直線上の3本以上の線も一度に結合可能**\n"
                    "- 距離制限なし：どの距離の壁でも結合可能\n"
                    "- 複数範囲同時処理：複数の結合を一度に実行\n"
                    "- 色分け表示：各選択範囲が異なる色で表示される"
                )
            elif edit_mode == "窓を追加して結合":
                st.markdown(
                    "**窓追加の手順:**\n"
                    "1. 下の画像上で**2回クリック**して矩形の対角を指定\n"
                    "2. 窓で分断された**2本の壁を矩形で囲む**\n"
                    "3. 「➕ この選択を追加」ボタンで追加（色が変わります）\n"
                    "4. 窓のサイズを入力:\n"
                    "   - 窓の高さ（m）: 例 1.2m\n"
                    "   - 床から窓下端までの高さ（m）: 例 0.9m\n"
                    "5. 「🪟 窓追加実行」で床側と天井側の壁を追加\n\n"
                    "**特徴:**\n"
                    "- 図面で窓により途切れた2本の壁を自動検出\n"
                    "- 床側の壁（床〜窓下端）と天井側の壁（窓上端〜天井）を追加\n"
                    "- 一条工務店の図面表記からそのまま入力可能"
                )
            elif edit_mode == "線を追加":
                st.markdown(
                    "**線追加の手順:**\n"
                    "1. 下の画像上で**2回クリック**して矩形の対角を指定\n"
                    "2. 追加したい線の位置を矩形で囲む\n"
                    "3. 「➕ この選択を追加」ボタンで追加（色が変わります）\n"
                    "4. さらに線を追加したければ手順1-3を繰り返す\n"
                    "5. 「➕ 線追加実行」で全ての選択範囲に線を追加\n\n"
                    "**方向の自動判定:**\n"
                    "- 矩形の幅 > 高さ → **横線**が中央に追加\n"
                    "- 矩形の高さ > 幅 → **縦線**が中央に追加\n"
                    "- 色分け表示：各選択範囲が異なる色で表示される"
                )
            elif edit_mode == "線を削除":
                st.markdown(
                    "**線削除の手順:**\n"
                    "1. 下の画像上で**2回クリック**して矩形の対角を指定\n"
                    "2. 削除したい壁線を**矩形で囲む**（複数本可）\n"
                    "3. 「➕ この選択を追加」ボタンで追加（色が変わります）\n"
                    "4. さらに削除対象を追加したければ手順1-3を繰り返す\n"
                    "5. 「🗑️ 削除実行」で矩形内の全ての壁を一括削除\n\n"
                    "**特徴:**\n"
                    "- **矩形で囲まれた全ての壁を一括削除**\n"
                    "- 複数箇所の壁をまとめて削除可能\n"
                    "- 矩形に完全に含まれる壁のみ対象"
                )
            else:  # スケール校正
                st.markdown(
                    "**スケール校正の手順:**\n"
                    "1. 下の画像上で**2回クリック**して矩形の対角を指定\n"
                    "2. 校正対象の1本の線を矩形で囲む\n"
                    "3. 囲んだ線のピクセル長を自動測定\n"
                    "4. その線が**何マス（90cm=1マス）**に相当するか入力\n"
                    "5. 「📏 スケール校正実行」でスケール値を自動計算\n"
                    "6. JSON・可視化画像が新しいスケールで再生成\n\n"
                    "**ポイント:**\n"
                    "- グリッドが見える状態で視覚的に確認できます\n"
                    "- 小数点での入力も可能（例：5.5マス）\n"
                    "- 校正後は新スケールで全ての3D座標が更新されます"
                )
        
        # セッションステートで矩形座標を管理
        if 'rect_coords' not in st.session_state:
            st.session_state.rect_coords = []
        if 'rect_coords_list' not in st.session_state:
            st.session_state.rect_coords_list = []  # 確定した矩形のリスト
        if 'reset_flag' not in st.session_state:
            st.session_state.reset_flag = False
        if 'last_click' not in st.session_state:
            st.session_state.last_click = None
        if 'merge_result' not in st.session_state:
            st.session_state.merge_result = None
        if 'edit_mode_state' not in st.session_state:
            st.session_state.edit_mode_state = "線を結合"  # 現在のモード
        
        # デバッグ情報：座標系パラメータを表示
        with st.expander("⚙️ 座標系情報（デバッグ）"):
            st.write(f"**可視化スケール (viz_scale):** {st.session_state.viz_scale} px/m")
            if st.session_state.json_bytes:
                try:
                    json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                    walls = json_data.get("walls", [])
                    if walls:
                        all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                        all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                        min_x, max_x = min(all_x), max(all_x)
                        min_y, max_y = min(all_y), max(all_y)
                        
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write(f"**X range:** {min_x:.3f}～{max_x:.3f} m")
                            st.write(f"**Y range:** {min_y:.3f}～{max_y:.3f} m")
                        with col_info2:
                            img_width = int((max_x - min_x) * st.session_state.viz_scale) + 100
                            img_height = int((max_y - min_y) * st.session_state.viz_scale) + 100
                            st.write(f"**予想画像サイズ:** {img_width} x {img_height} px")
                            st.write(f"**壁数:** {len(walls)}")
                except Exception as e:
                    st.error(f"座標系情報の取得エラー: {e}")
        
        # 矩形の色定義（OpenCV BGRフォーマット）
        RECT_COLORS = [
            (255, 0, 0),      # 赤
            (0, 255, 0),      # 緑
            (0, 0, 255),      # 青
            (255, 255, 0),    # 黄
            (255, 0, 255),    # マゼンタ
            (0, 255, 255),    # シアン
        ]
        
        # 距離閾値は無制限（非常に大きな値を設定）
        distance_threshold = 10000.0  # 無制限距離で結合
        
        # 編集結果の表示（セッション状態に保存されている場合）
        if st.session_state.merge_result is not None:
            result = st.session_state.merge_result
            
            st.success("🎉 編集完了！")
            st.markdown("### 📊 編集前後の比較")
            
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("**編集前**")
                st.image(Image.open(io.BytesIO(result['original_viz_bytes'])), use_container_width=True)
            with col_after:
                st.markdown("**編集後**")
                st.image(Image.open(io.BytesIO(result['edited_viz_bytes'])), use_container_width=True)
            
            # 統計情報の比較
            st.markdown("### 📈 統計")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("編集前の壁セグメント数", result['json_data']['metadata']['total_walls'])
            with col_stat2:
                st.metric(
                    "編集後の壁セグメント数", 
                    result['updated_json']['metadata']['total_walls'],
                    delta=result['updated_json']['metadata']['total_walls'] - result['json_data']['metadata']['total_walls']
                )
            
            # セッションステート更新の確認
            st.divider()
            col_save, col_discard = st.columns(2)
            with col_save:
                if st.button("💾 この結果を保存して続行", type="primary"):
                    # JSON・画像を更新
                    st.session_state.json_bytes = result['temp_json_path'].read_bytes()
                    st.session_state.json_name = "walls_3d_edited.json"
                    st.session_state.viz_bytes = result['temp_viz_path'].read_bytes()
                    
                    # 3DビューアHTMLも更新
                    st.session_state.viewer_html_bytes = result['viewer_html_bytes']
                    st.session_state.viewer_html_name = result['temp_viewer_path'].name
                    
                    # 状態を完全にクリア
                    st.session_state.rect_coords = []
                    st.session_state.rect_coords_list = []
                    st.session_state.last_click = None
                    st.session_state.reset_flag = False
                    st.session_state.merge_result = None
                    
                    st.success("✅ 保存しました。さらに編集を続けることができます。")
                    time.sleep(0.5)
                    st.rerun()
            with col_discard:
                if st.button("❌ この結果を破棄"):
                    # 元のJSON・画像を復元
                    original_json_data = result['json_data']
                    original_viz_bytes = result['original_viz_bytes']
                    
                    # JSONを一時ファイルに書き戻す
                    temp_json_path = Path(st.session_state.out_dir) / "walls_3d_edited.json"
                    with open(temp_json_path, 'w', encoding='utf-8') as f:
                        json.dump(original_json_data, f, indent=2, ensure_ascii=False)
                    
                    # セッション状態を元に戻す
                    st.session_state.json_bytes = temp_json_path.read_bytes()
                    st.session_state.viz_bytes = original_viz_bytes
                    
                    # 状態をクリア
                    st.session_state.rect_coords = []
                    st.session_state.rect_coords_list = []
                    st.session_state.last_click = None
                    st.session_state.reset_flag = False
                    st.session_state.merge_result = None
                    st.info("✅ 編集を破棄して元に戻しました。")
                    st.rerun()
        else:
            # 編集結果がない場合のみ、編集UIを表示
            
            # 照明配置モードの場合は独自のUIを表示
            if edit_mode == "照明を配置":
                # 照明配置モード
                st.markdown(
                    "可視化画像上で**クリック**してスポットライトを配置します。"
                )
                
                # JSON初期化（初回のみ）
                if 'lights_list' not in st.session_state:
                    json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                    st.session_state.lights_list = json_data['metadata'].get('lights', [])
                
                # 最後のクリック位置を記録（重複防止）
                if 'last_light_click' not in st.session_state:
                    st.session_state.last_light_click = None
                
                # 画像を読み込み
                viz_image = Image.open(io.BytesIO(st.session_state.viz_bytes))
                
                # 画像を固定幅にリサイズ
                if viz_image.width != DISPLAY_IMAGE_WIDTH:
                    scale_ratio = DISPLAY_IMAGE_WIDTH / viz_image.width
                    new_height = int(viz_image.height * scale_ratio)
                    viz_image_resized = viz_image.resize((DISPLAY_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
                else:
                    viz_image_resized = viz_image
                    scale_ratio = 1.0
                
                # 画像表示とクリック処理
                click_result = streamlit_image_coordinates(
                    np.array(viz_image_resized),
                    key=f"light_placement_{st.session_state.viz_name}"
                )
                
                if click_result is not None:
                    # クリック座標を取得
                    click_x = click_result.get('x')
                    click_y = click_result.get('y')
                    
                    if click_x is not None and click_y is not None:
                        # 元の画像サイズに戻す
                        actual_x = int(click_x / scale_ratio)
                        actual_y = int(click_y / scale_ratio)
                        
                        # ピクセル座標をメートル座標に変換
                        json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                        walls = json_data['walls']
                        all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                        all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                        min_x, min_y = min(all_x), min(all_y)
                        scale = int(st.session_state.viz_scale)
                        margin = 50
                        
                        # ピクセル→メートル変換（元の画像サイズ基準）
                        meter_x = (actual_x - margin) / scale + min_x
                        meter_y = (viz_image.height - actual_y - margin) / scale + min_y
                        
                        # デバッグ情報
                        st.write(f"**デバッグ**: クリック位置(px)=({click_x}, {click_y}) → 実座標(px)=({actual_x}, {actual_y})")
                        st.write(f"**デバッグ**: 画像サイズ={viz_image.width}x{viz_image.height}px, scale={scale}px/m, margin={margin}px")
                        st.write(f"**デバッグ**: 座標範囲: X=[{min_x:.2f}, {max(all_x):.2f}]m, Y=[{min_y:.2f}, {max(all_y):.2f}]m")
                        st.write(f"**デバッグ**: 変換後のメートル座標: X={meter_x:.3f}m, Y={meter_y:.3f}m")
                        
                        # 重複クリック防止（同じ座標のクリックを無視）
                        current_click = (click_x, click_y)
                        if st.session_state.last_light_click != current_click:
                            # 照明を追加（高さは天井高さの80%）
                            avg_wall_height = sum(w['height'] for w in walls) / len(walls) if walls else 2.7
                            light_z = avg_wall_height * 0.8
                            
                            new_light = {
                                'position': [round(meter_x, 3), round(light_z, 3), round(meter_y, 3)],
                                'intensity': 1.0,
                                'color': '#ffffff'
                            }
                            st.session_state.lights_list.append(new_light)
                            st.session_state.last_light_click = current_click
                            st.success(f"✅ 照明を追加しました (位置: [X={meter_x:.3f}m, Z={light_z:.3f}m, Y={meter_y:.3f}m])")
                            st.rerun()
                
                # 照明リスト表示
                st.markdown("### 配置済み照明")
                if st.session_state.lights_list:
                    col_clear, col_count = st.columns([1, 3])
                    with col_clear:
                        if st.button("🗑️ 全削除"):
                            st.session_state.lights_list = []
                            st.session_state.last_light_click = None
                            st.rerun()
                    with col_count:
                        st.write(f"合計: {len(st.session_state.lights_list)} 個")
                    
                    for i, light in enumerate(st.session_state.lights_list):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            pos = light['position']
                            st.write(f"照明 {i+1}: [X={pos[0]:.3f}m, Y(高)={pos[1]:.3f}m, Z={pos[2]:.3f}m]")
                        with col2:
                            if st.button("削除", key=f"delete_light_{i}"):
                                st.session_state.lights_list.pop(i)
                                st.rerun()
                else:
                    st.info("照明はまだ配置されていません")
                
                # 保存ボタン
                if st.button("💾 照明設定を保存", type="primary"):
                    with st.spinner("照明設定を保存中..."):
                        try:
                            st.write("🔧 デバッグ: 保存処理開始")
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            json_data['metadata']['lights'] = st.session_state.lights_list
                            
                            st.write(f"🔧 デバッグ: 照明数={len(st.session_state.lights_list)}")
                            
                            out_dir = Path(st.session_state.out_dir)
                            json_path = out_dir / st.session_state.json_name
                            
                            st.write(f"🔧 デバッグ: JSONパス={json_path}")
                            
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)
                            
                            st.write("🔧 デバッグ: JSON保存完了")
                            
                            # 照明付き3DビューアHTMLを別名で生成
                            viewer_html_lights = out_dir / "viewer_3d_with_lights.html"
                            
                            st.write(f"🔧 デバッグ: HTMLパス={viewer_html_lights}")
                            
                            _generate_3d_viewer_html(json_path, viewer_html_lights, with_lights=True)
                            
                            st.write("🔧 デバッグ: HTML生成完了")
                            
                            # セッション状態を更新
                            st.session_state.json_bytes = json_path.read_bytes()
                            st.session_state.viewer_html_lights_bytes = viewer_html_lights.read_bytes()
                            st.session_state.viewer_html_lights_name = viewer_html_lights.name
                            
                            st.success("✅ 照明設定を保存しました")
                            st.info(f"📊 照明付き3Dビューア: {viewer_html_lights.name} を生成しました")
                            st.write("🔄 画面を更新します...")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ エラー: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                # リセットボタンと追加ボタン（先に配置）
                col_reset, col_add, col_exec = st.columns(3)
                with col_reset:
                    if st.button("🗑️ 選択リセット"):
                        st.session_state.rect_coords = []
                        st.session_state.rect_coords_list = []
                        st.session_state.reset_flag = True
                        st.session_state.last_click = None
                        st.session_state.merge_result = None  # 結合結果もクリア
                        st.rerun()
            
            # 可視化画像を読み込み（照明配置モード以外）
            if st.session_state.viz_bytes and edit_mode != "照明を配置":
                viz_img = Image.open(io.BytesIO(st.session_state.viz_bytes))
                
                # 選択範囲を描画した画像を作成
                import cv2
                display_img_array = np.array(viz_img.copy())
                
                # 確定済みの矩形を描画（異なる色で）
                for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                    color = RECT_COLORS[idx % len(RECT_COLORS)]
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    # 半透明の矩形を描画
                    overlay = display_img_array.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.25, display_img_array, 0.75, 0, display_img_array)
                    # 矩形の枠線を描画
                    cv2.rectangle(display_img_array, (x1, y1), (x2, y2), color, 3)
                    # 番号を描画
                    cv2.putText(display_img_array, f"{idx+1}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 現在選択中の矩形を描画（次の色で表示）
                if len(st.session_state.rect_coords) == 1:
                    # 1点目を円で表示（次の色）
                    next_color = RECT_COLORS[len(st.session_state.rect_coords_list) % len(RECT_COLORS)]
                    cv2.circle(display_img_array, st.session_state.rect_coords[0], 10, next_color, -1)
                    cv2.circle(display_img_array, st.session_state.rect_coords[0], 12, next_color, 2)
                elif len(st.session_state.rect_coords) == 2:
                    # 2点で矩形を描画（次の色）
                    next_color = RECT_COLORS[len(st.session_state.rect_coords_list) % len(RECT_COLORS)]
                    p1, p2 = st.session_state.rect_coords
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    # 半透明の矩形を描画
                    overlay = display_img_array.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), next_color, -1)
                    cv2.addWeighted(overlay, 0.3, display_img_array, 0.7, 0, display_img_array)
                    # 矩形の枠線を描画
                    cv2.rectangle(display_img_array, (x1, y1), (x2, y2), next_color, 3)
            
                display_img = Image.fromarray(display_img_array)
            
                # 画像を固定幅にリサイズ
                if display_img.width != DISPLAY_IMAGE_WIDTH:
                    scale_ratio = DISPLAY_IMAGE_WIDTH / display_img.width
                    new_height = int(display_img.height * scale_ratio)
                    display_img_resized = display_img.resize((DISPLAY_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
                else:
                    display_img_resized = display_img
                    scale_ratio = 1.0
            
                # UI表示：モード別
                if edit_mode == "線を削除":
                    # 削除モード：2点選択で矩形を指定
                    if len(st.session_state.rect_coords) == 1:
                        st.info(f"✓ 1点目選択: ({st.session_state.rect_coords[0][0]}, {st.session_state.rect_coords[0][1]})")
                    elif len(st.session_state.rect_coords) == 2:
                        p1, p2 = st.session_state.rect_coords
                        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})")
                    st.write("画像をクリックして矩形の2点を指定してください（1点目→2点目）")
                elif edit_mode == "スケール校正":
                    # スケール校正モード：2点選択で線を囲む
                    if len(st.session_state.rect_coords) == 1:
                        st.info(f"✓ 1点目選択: ({st.session_state.rect_coords[0][0]}, {st.session_state.rect_coords[0][1]})")
                    elif len(st.session_state.rect_coords) == 2:
                        p1, p2 = st.session_state.rect_coords
                        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        px_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})\n線の長さ: {px_distance:.1f}px")
                    st.write("画像をクリックして矩形の2点を指定してください（1点目→2点目）")
                else:
                    # 結合・追加モード：2点選択
                    if len(st.session_state.rect_coords) == 1:
                        st.info(f"✓ 1点目選択: ({st.session_state.rect_coords[0][0]}, {st.session_state.rect_coords[0][1]})")
                    elif len(st.session_state.rect_coords) == 2:
                        p1, p2 = st.session_state.rect_coords
                        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][len(st.session_state.rect_coords_list) % 6]
                        st.success(f"✅ 2点選択完了（{color_name}）: ({x1}, {y1}) - ({x2}, {y2})")
                    st.write("画像をクリックして矩形の2点を指定してください（1点目→2点目）")
            
                # クリック可能な画像を表示（キーを動的に変更して値をリセット）
                coord_key = f"image_coords_{len(st.session_state.rect_coords_list)}_{len(st.session_state.rect_coords)}"
            
                st.markdown(
                    """
                    <p style="font-size: 12px; color: #666; margin-bottom: 8px;">
                    💡 <b>注:</b> 画像が見切れる場合は、ブラウザの画面スケール（Ctrl/Cmd + マイナスキー）を小さくしてください。
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            
                # 画像を元のサイズで表示（リサイズなし）
                value = streamlit_image_coordinates(
                    display_img_resized,
                    key=coord_key
                )
                
                # リサイズ時の座標変換
                if value is not None and value.get("x") is not None and scale_ratio != 1.0:
                    # 元の座標に変換
                    value["x"] = int(value["x"] / scale_ratio)
                    value["y"] = int(value["y"] / scale_ratio)

                # デバッグ: クリック座標を表示
                if value is not None and value.get("x") is not None:
                    st.caption(
                        f"クリック座標: raw=({value['x']}, {value['y']}) | "
                        f"表示画像サイズ={display_img_resized.width}x{display_img_resized.height}px | "
                        f"scale_ratio={scale_ratio:.3f}"
                    )
            
                # クリックされた座標を記録（重複チェック）
                if value is not None and value.get("x") is not None:
                    new_point = (value["x"], value["y"])
                
                    if edit_mode == "線を削除":
                        # 削除モード：2点選択で矩形
                        if len(st.session_state.rect_coords) < 2:
                            if len(st.session_state.rect_coords) == 0 or st.session_state.last_click != new_point:
                                st.session_state.rect_coords.append(new_point)
                                st.session_state.last_click = new_point
                                st.rerun()
                    else:
                        # 結合・追加モード：2点選択
                        if len(st.session_state.rect_coords) < 2:
                            if len(st.session_state.rect_coords) == 0 or st.session_state.last_click != new_point:
                                st.session_state.rect_coords.append(new_point)
                                st.session_state.last_click = new_point
                                st.rerun()  # 画像を再描画して選択点を表示
            
                # 選択完了時のUI
                if edit_mode == "線を削除" and len(st.session_state.rect_coords) == 2:
                    # 削除モード：2点選択完了（矩形）
                    p1, p2 = st.session_state.rect_coords
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})")
                
                    with col_add:
                        if st.button("➕ この選択を追加", type="primary"):
                            # 現在の2点をリストに追加
                            st.session_state.rect_coords_list.append((p1, p2))
                            # 現在の選択をクリア
                            st.session_state.rect_coords = []
                            st.session_state.last_click = None
                            st.rerun()
                elif edit_mode == "スケール校正" and len(st.session_state.rect_coords) == 2:
                    # スケール校正モード：2点選択完了（1本の線を囲む）
                    p1, p2 = st.session_state.rect_coords
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    px_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    st.success(f"✅ 2点選択完了: 線の長さ = {px_distance:.1f}px")
                
                    with col_add:
                        if st.button("➕ この選択を追加", type="primary"):
                            # スケール校正用に選択を保存（1つだけ）
                            st.session_state.rect_coords_list = [(p1, p2)]  # 最新の選択のみ保持
                            st.session_state.rect_coords = []
                            st.session_state.last_click = None
                            st.rerun()
                elif edit_mode != "線を削除" and edit_mode != "スケール校正" and len(st.session_state.rect_coords) == 2:
                    # 結合・追加・窓追加モード：2点選択完了
                    p1, p2 = st.session_state.rect_coords
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})")
                    
                    # 窓追加モードの場合、選択範囲内の壁を表示
                    if edit_mode == "窓を追加して結合":
                        # 現在のJSONデータから壁を検出
                        try:
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls = json_data['walls']
                            
                            # 可視化画像のパラメータを取得
                            all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                            all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)
                            
                            scale = int(viz_scale)
                            margin = 50
                            img_width = int((max_x - min_x) * scale) + 2 * margin
                            img_height = int((max_y - min_y) * scale) + 2 * margin
                            
                            rect = {
                                'left': x1,
                                'top': y1,
                                'width': x2 - x1,
                                'height': y2 - y1
                            }
                            
                            walls_in_rect = _filter_walls_strictly_in_rect(
                                walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                            )
                            
                            if len(walls_in_rect) == 2:
                                st.info(f"🎯 この範囲に2本の壁が検出されました（ID: {walls_in_rect[0]['id']}, {walls_in_rect[1]['id']}）")
                            elif len(walls_in_rect) < 2:
                                st.warning(f"⚠️ この範囲に{len(walls_in_rect)}本の壁しか検出されません（2本必要）")
                            else:
                                st.warning(f"⚠️ この範囲に{len(walls_in_rect)}本の壁が検出されました（2本のみ選択してください）")
                        except Exception as e:
                            st.error(f"壁検出エラー: {e}")
                
                    with col_add:
                        if st.button("➕ この選択を追加", type="primary"):
                            # 現在の2点をリストに追加
                            st.session_state.rect_coords_list.append((p1, p2))
                            # 現在の選択をクリア
                            st.session_state.rect_coords = []
                            st.session_state.last_click = None
                            st.rerun()
            
                # 確定済み選択の表示
                if len(st.session_state.rect_coords_list) > 0:
                    if edit_mode == "線を削除":
                        st.markdown("### 📋 追加済みの削除対象")
                        for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                            color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][idx % 6]
                            st.write(f"#{idx+1}（{color_name}）: ({p1[0]}, {p1[1]})")
                    elif edit_mode == "窓を追加して結合":
                        st.markdown("### 📋 追加済みの選択範囲（窓）")
                        # 窓追加モードの場合、各選択範囲の壁検出状況を表示
                        try:
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls = json_data['walls']
                            all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                            all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)
                            scale = int(viz_scale)
                            margin = 50
                            img_width = int((max_x - min_x) * scale) + 2 * margin
                            img_height = int((max_y - min_y) * scale) + 2 * margin
                            
                            for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                                color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][idx % 6]
                                
                                rect = {'left': x1, 'top': y1, 'width': x2 - x1, 'height': y2 - y1}
                                walls_in_rect = _filter_walls_strictly_in_rect(
                                    walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                )
                                
                                status = ""
                                if len(walls_in_rect) == 2:
                                    status = f"✅ 2本検出（ID: {walls_in_rect[0]['id']}, {walls_in_rect[1]['id']}）"
                                elif len(walls_in_rect) < 2:
                                    status = f"❌ {len(walls_in_rect)}本のみ"
                                else:
                                    status = f"⚠️ {len(walls_in_rect)}本（多すぎ）"
                                
                                st.write(f"#{idx+1}（{color_name}）: ({x1}, {y1}) - ({x2}, {y2}) {status}")
                        except Exception as e:
                            st.error(f"壁検出エラー: {e}")
                    else:
                        st.markdown("### 📋 追加済みの選択範囲")
                        for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                            color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][idx % 6]
                            st.write(f"#{idx+1}（{color_name}）: ({x1}, {y1}) - ({x2}, {y2})")
            
                with col_exec:
                    # モード別のボタン表示と処理
                    if edit_mode == "線を結合":
                        button_label = "🔗 結合実行"
                    elif edit_mode == "窓を追加して結合":
                        button_label = "🪟 窓追加実行"
                    elif edit_mode == "線を追加":
                        button_label = "➕ 線追加実行"
                    elif edit_mode == "スケール校正":
                        button_label = "📏 スケール校正実行"
                    else:  # 線を削除
                        button_label = "🗑️ 削除実行"
                
                    if edit_mode == "窓を追加して結合":
                        # 窓追加：選択範囲があれば窓サイズ入力を表示
                        if len(st.session_state.rect_coords_list) > 0:
                            st.markdown("### 🪟 窓のサイズを入力")
                            
                            # 既存の壁から天井高さを取得
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls = json_data['walls']
                            heights = [w.get('height', 2.4) for w in walls if 'height' in w]
                            default_room_height = min(max(heights) if heights else 2.4, 10.0)  # 10.0mでクリップ
                            
                            col_w1, col_w2, col_w3 = st.columns(3)
                            with col_w1:
                                window_height = st.number_input(
                                    "窓の高さ（m）",
                                    min_value=0.1,
                                    max_value=3.0,
                                    value=1.2,
                                    step=0.1,
                                    help="窓の上下方向のサイズ"
                                )
                            with col_w2:
                                base_height = st.number_input(
                                    "床から窓下端まで（m）",
                                    min_value=0.0,
                                    max_value=5.0,
                                    value=0.9,
                                    step=0.1,
                                    help="床面から窓の下端までの高さ"
                                )
                            with col_w3:
                                room_height = st.number_input(
                                    "部屋の天井高さ（m）",
                                    min_value=1.0,
                                    max_value=10.0,
                                    value=default_room_height,
                                    step=0.1,
                                    help="床から天井までの高さ"
                                )
                            
                            # 計算結果を表示
                            ceiling_height = room_height - (base_height + window_height)
                            st.info(f"📐 床側の壁: {base_height:.2f}m、天井側の壁: {ceiling_height:.2f}m")
                            
                            if ceiling_height < 0:
                                st.error("⚠️ 窓のサイズが部屋の高さを超えています")
                            
                            # 実行ボタンを表示
                            execute_window_addition = st.button(button_label, type="primary")
                            if execute_window_addition:
                                # 窓追加処理のパラメータをセッション状態に保存
                                st.session_state.window_execution_params = {
                                    'window_height': window_height,
                                    'base_height': base_height,
                                    'room_height': room_height
                                }
                                st.session_state.execute_window_now = True
                    
                    elif edit_mode == "スケール校正":
                        # スケール校正：1つの選択のみ使用
                        if len(st.session_state.rect_coords_list) == 1:
                            # グリッド数入力
                            p1, p2 = st.session_state.rect_coords_list[0]
                            rect_x1, rect_y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                            rect_x2, rect_y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        
                            st.write(f"**デバッグ情報（矩形座標）:** ({rect_x1}, {rect_y1}) - ({rect_x2}, {rect_y2})")
                        
                            # JSONから壁データを取得
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls = json_data['walls']
                        
                            st.write(f"**デバッグ: JSON内の壁数 = {len(walls)}本**")
                        
                            # メートル→ピクセル変換の基準となる値を一度だけ計算
                            scale = int(viz_scale)
                            margin = 50
                            all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                            all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)
                        
                            # 画像高さを計算（Y座標反転用）
                            img_width = int((max_x - min_x) * scale) + 2 * margin
                            img_height = int((max_y - min_y) * scale) + 2 * margin
                        
                            # 矩形内に完全に含まれる壁線分を検出
                            walls_in_rect = []
                            for wall in walls:
                                start_m = wall['start']  # メートル単位
                                end_m = wall['end']
                            
                                # メートル→ピクセル変換（Y座標反転を考慮）
                                start_px_x = int((start_m[0] - min_x) * scale) + margin
                                start_px_y = img_height - (int((start_m[1] - min_y) * scale) + margin)
                                end_px_x = int((end_m[0] - min_x) * scale) + margin
                                end_px_y = img_height - (int((end_m[1] - min_y) * scale) + margin)
                            
                                # 誤差範囲（ピクセル単位で±5px）を考慮した判定
                                tolerance = 5
                                start_in_rect = (rect_x1 - tolerance <= start_px_x <= rect_x2 + tolerance and
                                               rect_y1 - tolerance <= start_px_y <= rect_y2 + tolerance)
                                end_in_rect = (rect_x1 - tolerance <= end_px_x <= rect_x2 + tolerance and
                                             rect_y1 - tolerance <= end_px_y <= rect_y2 + tolerance)
                            
                                # 線の両端が矩形内に含まれるかチェック
                                if start_in_rect and end_in_rect:
                                    wall_length_px = math.sqrt((end_px_x - start_px_x)**2 + (end_px_y - start_px_y)**2)
                                    walls_in_rect.append({
                                        'wall': wall,
                                        'id': wall['id'],
                                        'start_px': (start_px_x, start_px_y),
                                        'end_px': (end_px_x, end_px_y),
                                        'length_px': wall_length_px,
                                        'start_m': start_m,
                                        'end_m': end_m
                                    })
                                    st.write(f"✅ 壁({wall['id']}) を検出 - ピクセル長: {wall_length_px:.1f}px")
                        
                            st.write(f"**検出結果:** {len(walls_in_rect)} 個の壁が矩形内に見つかりました")
                        
                            if walls_in_rect:
                                # 最長の壁を使用（複数壁がある場合）
                                target_wall_data = max(walls_in_rect, key=lambda w: w['length_px'])
                                px_distance = target_wall_data['length_px']
                                wall_id = target_wall_data['id']
                            
                                st.markdown("---")
                                st.markdown("### 🔧 スケール値計算")
                                st.info(f"✅ 矩形内で壁({wall_id})を検出しました")
                                st.write(f"**測定した壁の長さ**: {px_distance:.1f}px")
                            else:
                                st.error("❌ 矩形内に完全に含まれる壁が見つかりません。もう一度選択してください。")
                                st.write("**トラブルシューティング:**")
                                st.write("- 矩形の2点がピクセル座標系とメートル座標系でズレている可能性")
                                st.write("- 表示画像がリサイズされている場合、座標変換が必要な場合があります")
                                st.write("- 矩形を拡大してからもう一度お試しください")
                                px_distance = None
                        
                            if px_distance is not None:
                                grid_cells = st.number_input(
                                    "この壁が何マス（90cm=1マス）に相当するか:",
                                    min_value=0.1,
                                    value=1.0,
                                    step=0.1,
                                    help="グリッド線1本分 = 1マス = 90cm"
                                )
                            
                                if st.button(button_label, type="primary"):
                                    # 新しいpixel_to_meterを計算
                                    actual_distance_m = grid_cells * 0.45  # 1マス = 0.45m = 45cm
                                    new_pixel_to_meter = actual_distance_m / px_distance
                                
                                    st.success(f"新しいpixel_to_meter: {new_pixel_to_meter:.6f}")
                                    st.info(f"({grid_cells}マス = {actual_distance_m:.2f}m / {px_distance:.1f}px)")
                                
                                    try:
                                        # 現在のJSONを読み込み（編集済みの状態を保持）
                                        import copy
                                        current_json = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                        old_pixel_to_meter = current_json['metadata'].get('pixel_to_meter', pixel_to_meter)
                                        
                                        # スケール変換比率を計算
                                        scale_ratio = new_pixel_to_meter / old_pixel_to_meter
                                        
                                        # 各壁の座標をスケール変換
                                        calibrated_json = copy.deepcopy(current_json)
                                        for wall in calibrated_json['walls']:
                                            # 座標をスケール変換
                                            wall['start'] = [round(wall['start'][0] * scale_ratio, 3), 
                                                           round(wall['start'][1] * scale_ratio, 3)]
                                            wall['end'] = [round(wall['end'][0] * scale_ratio, 3), 
                                                         round(wall['end'][1] * scale_ratio, 3)]
                                            # 長さを再計算
                                            dx = wall['end'][0] - wall['start'][0]
                                            dy = wall['end'][1] - wall['start'][1]
                                            wall['length'] = round(math.sqrt(dx**2 + dy**2), 3)
                                            
                                            # 壁の高さと厚さにも倍率を適用（マス目：厚さ：高さの比率を保持）
                                            wall['height'] = round(wall['height'] * scale_ratio, 3)
                                            wall['thickness'] = round(wall['thickness'] * scale_ratio, 3)
                                        
                                        # メタデータを更新
                                        calibrated_json['metadata']['pixel_to_meter'] = new_pixel_to_meter
                                        
                                        # 新しいJSONパスと可視化パス
                                        out_dir = Path(st.session_state.out_dir)
                                        json_path_new = out_dir / "walls_3d_calibrated.json"
                                        viz_path_new = out_dir / "visualization_calibrated.png"
                                        blender_path_new = out_dir / "import_walls_calibrated.py"
                                        
                                        # JSONを保存
                                        with open(json_path_new, 'w', encoding='utf-8') as f:
                                            json.dump(calibrated_json, f, indent=2, ensure_ascii=False)
                                        
                                        # 可視化を再生成
                                        visualize_3d_walls(str(json_path_new), str(viz_path_new), scale=int(viz_scale))
                                        
                                        # Blenderスクリプトも再生成
                                        _generate_blender_script(json_path_new, blender_path_new)
                                        
                                        # セッション状態を更新
                                        st.session_state.json_bytes = json_path_new.read_bytes()
                                        st.session_state.json_name = json_path_new.name
                                        st.session_state.viz_bytes = viz_path_new.read_bytes()
                                        st.session_state.viz_name = viz_path_new.name
                                        st.session_state.blender_bytes = blender_path_new.read_bytes()
                                        st.session_state.blender_name = blender_path_new.name
                                        
                                        st.success("✅ JSON・可視化・Blenderスクリプトを再生成しました！")
                                        st.info(f"📝 編集済みの壁構成を維持したまま、スケールのみを調整しました")
                                        
                                        # リセット
                                        st.session_state.rect_coords_list = []
                                        st.session_state.rect_coords = []
                                        st.session_state.last_click = None
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ エラーが発生しました: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())
                
                    # 窓追加モードで実行ボタンが押された場合の処理
                    if edit_mode == "窓を追加して結合" and st.session_state.get('execute_window_now'):
                        # フラグをクリア
                        st.session_state.execute_window_now = False
                        
                        try:
                            st.info("🔄 窓追加処理を開始します...")
                            
                            # 処理対象の矩形リストを作成
                            target_rects = list(st.session_state.rect_coords_list)
                            if len(st.session_state.rect_coords) == 2:
                                target_rects.append(tuple(st.session_state.rect_coords))
                        
                            # JSONデータを読み込み
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls = json_data['walls']
                        
                            # 可視化画像のパラメータを取得
                            all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                            all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)
                        
                            scale = int(viz_scale)
                            margin = 50
                            img_width = int((max_x - min_x) * scale) + 2 * margin
                            img_height = int((max_y - min_y) * scale) + 2 * margin
                        
                            # 編集前の画像を保存（比較用）
                            original_viz_bytes = st.session_state.viz_bytes
                        
                            # 元データを保護するためディープコピー
                            import copy
                            original_json_data = copy.deepcopy(json_data)
                            updated_json = json_data
                            
                            # 追加した壁のID（赤色表示用）
                            added_wall_ids = []
                            
                            # ===== 窓を追加して結合モード =====
                            # デバッグ情報を表示するエクスパンダー
                            with st.expander("🔍 窓追加処理のデバッグ情報", expanded=True):
                                # セッションステートからパラメータを取得
                                params = st.session_state.get('window_execution_params', {})
                                window_height = params.get('window_height', 1.2)
                                base_height = params.get('base_height', 0.9)
                                room_height = params.get('room_height', 2.4)
                                
                                st.info(f"📝 窓パラメータ: 高さ={window_height}m, 床から={base_height}m, 天井={room_height}m")
                                st.info(f"📦 処理対象矩形数: {len(target_rects)}個")
                                
                                total_added_count = 0
                                window_details = []
                
                                for rect_idx, (p1, p2) in enumerate(target_rects):
                                    st.markdown(f"---\n**矩形#{rect_idx+1}の処理:**")
                                    rect = {
                                        'left': min(p1[0], p2[0]),
                                        'top': min(p1[1], p2[1]),
                                        'width': abs(p2[0] - p1[0]),
                                        'height': abs(p2[1] - p1[1])
                                    }
                                
                                    # 矩形内に完全に含まれる壁線を抽出（2本を期待）
                                    walls_in_rect = _filter_walls_strictly_in_rect(
                                        updated_json['walls'], rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                    )
                                
                                    st.write(f"**選択範囲内の壁:** {len(walls_in_rect)}本")
                                    if walls_in_rect:
                                        st.write(f"検出された壁ID: {[w['id'] for w in walls_in_rect]}")
                                
                                    if len(walls_in_rect) == 2:
                                        # 2本の壁の間に床側と天井側の壁を追加
                                        st.success(f"✅ 2本の壁を検出、窓追加処理を実行します")
                                        updated_json, added_walls = _add_window_walls(
                                            updated_json,
                                            walls_in_rect[0],
                                            walls_in_rect[1],
                                            window_height,
                                            base_height,
                                            room_height
                                        )
                                        total_added_count += len(added_walls)
                                        st.success(f"✅ {len(added_walls)}本の壁を追加しました（ID: {[w['id'] for w in added_walls]}）")
                                        
                                        # 追加した壁のIDを記録（赤色表示用）
                                        added_wall_ids.extend([w['id'] for w in added_walls])
                                    
                                        color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                        window_details.append({
                                            'rect_idx': rect_idx,
                                            'color_name': color_name,
                                            'wall_ids': [w['id'] for w in added_walls],
                                            'window_height': window_height,
                                            'base_height': base_height
                                        })
                                    elif len(walls_in_rect) < 2:
                                        st.warning(f"⚠️ 矩形#{rect_idx+1}: 2本の壁が必要ですが、{len(walls_in_rect)}本しか見つかりません")
                                    else:
                                        st.warning(f"⚠️ 矩形#{rect_idx+1}: 2本の壁を選択してください（{len(walls_in_rect)}本選択されています）")
                
                                if total_added_count > 0:
                                    st.success(f"✅✅ 合計 {total_added_count} 本の壁を追加しました（窓{len(window_details)}箇所）")
                                
                                    # 追加詳細を表示
                                    st.markdown("**窓追加結果:**")
                                    for detail in window_details:
                                        st.write(
                                            f"#{detail['rect_idx']+1}（{detail['color_name']}）: "
                                            f"壁({detail['wall_ids'][0]}, {detail['wall_ids'][1]}) を追加 - "
                                            f"窓高さ: {detail['window_height']}m, 床から: {detail['base_height']}m"
                                        )
                                else:
                                    st.warning("⚠️ 追加可能な窓が見つかりません")
                            
                            # 一時ファイルに保存
                            temp_json_path = Path(st.session_state.out_dir) / "walls_3d_edited.json"
                            with open(temp_json_path, 'w', encoding='utf-8') as f:
                                json.dump(updated_json, f, indent=2, ensure_ascii=False)
                            
                            # 再可視化（元の変換と同じスケールを使用）
                            # 追加した壁を赤色で表示
                            temp_viz_path = Path(st.session_state.out_dir) / "visualization_edited.png"
                            visualize_3d_walls(str(temp_json_path), str(temp_viz_path), scale=int(viz_scale), highlight_wall_ids=added_wall_ids)
                        
                            # 3Dビューア生成
                            temp_viewer_path = Path(st.session_state.out_dir) / "viewer_3d_edited.html"
                            _generate_3d_viewer_html(temp_json_path, temp_viewer_path)
                        
                            # セッション状態を更新
                            st.session_state.json_bytes = temp_json_path.read_bytes()
                            st.session_state.viz_bytes = temp_viz_path.read_bytes()
                        
                            # 編集後の画像を読み込み
                            edited_viz_bytes = temp_viz_path.read_bytes()
                            viewer_html_bytes = temp_viewer_path.read_bytes()
                        
                            # 編集結果をセッション状態に保存
                            edit_count = total_added_count
                            edit_details = window_details
                        
                            st.session_state.merge_result = {
                                'original_viz_bytes': original_viz_bytes,
                                'edited_viz_bytes': edited_viz_bytes,
                                'json_data': original_json_data,
                                'updated_json': updated_json,
                                'temp_json_path': temp_json_path,
                                'temp_viz_path': temp_viz_path,
                                'temp_viewer_path': temp_viewer_path,
                                'viewer_html_bytes': viewer_html_bytes,
                                'edit_count': edit_count,
                                'edit_details': edit_details
                            }
                            # 編集状態をリセット
                            st.session_state.rect_coords = []
                            st.session_state.rect_coords_list = []
                            # 窓追加パラメータもクリア
                            if 'window_execution_params' in st.session_state:
                                del st.session_state.window_execution_params
                            
                            st.success("✅ 処理完了！結果画面に移動します...")
                            time.sleep(1)  # デバッグ情報を確認できるように1秒待機
                            st.rerun()
                    
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    elif len(st.session_state.rect_coords_list) > 0 or len(st.session_state.rect_coords) == 2:
                        # 結合・追加・削除モードの実行ボタン（窓追加モードは上記で別処理）
                        should_execute = False
                        if st.button(button_label, type="primary"):
                            should_execute = True
                        
                        if should_execute:
                            try:
                                # 処理対象の矩形リストを作成（確定済み選択 + 現在選択中の2点）
                                target_rects = list(st.session_state.rect_coords_list)
                                if len(st.session_state.rect_coords) == 2:
                                    target_rects.append(tuple(st.session_state.rect_coords))
                            
                                # JSONデータを読み込み
                                json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                walls = json_data['walls']
                            
                                # 可視化画像のパラメータを取得
                                all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                                all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                                min_x, max_x = min(all_x), max(all_x)
                                min_y, max_y = min(all_y), max(all_y)
                            
                                scale = int(viz_scale)
                                margin = 50
                                img_width = int((max_x - min_x) * scale) + 2 * margin
                                img_height = int((max_y - min_y) * scale) + 2 * margin
                            
                                # 編集前の画像を保存（比較用）
                                original_viz_bytes = st.session_state.viz_bytes
                            
                                # 元データを保護するためディープコピー
                                import copy
                                original_json_data = copy.deepcopy(json_data)
                                updated_json = json_data
                                
                                # 追加した壁のID（窓追加モードで使用）
                                added_wall_ids = []
                            
                                # 各モード用の変数を事前初期化
                                total_merged_count = 0
                                merge_details = []
                                total_added_count = 0
                                add_details = []
                                total_deleted_count = 0
                                delete_details = []
                                
                                if edit_mode == "線を結合":
                                    # ===== 線を結合モード =====
                                    # 各矩形をループして処理
                                    total_merged_count = 0
                                    merge_details = []
                                
                                    for rect_idx, (p1, p2) in enumerate(target_rects):
                                        rect = {
                                            'left': min(p1[0], p2[0]),
                                            'top': min(p1[1], p2[1]),
                                            'width': abs(p2[0] - p1[0]),
                                            'height': abs(p2[1] - p1[1])
                                        }
                                    
                                        # 選択範囲内の壁線を抽出（精密フィルタリング：完全に含まれるもののみ）
                                        walls_in_selection = _filter_walls_strictly_in_rect(
                                            updated_json['walls'], rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                        )
                                    
                                        st.write(f"**選択範囲内の壁:** {len(walls_in_selection)}本")
                                        if walls_in_selection:
                                            wall_ids_in_selection = [w['id'] for w in walls_in_selection]
                                            wall_display = ", ".join([f"壁({wid})" for wid in wall_ids_in_selection])
                                            st.write(f"壁: {wall_display}")
                                    
                                        if len(walls_in_selection) >= 2:
                                            # 複数線が選択されている場合、方向を判定して最も離れた2本のペアのみを結合
                                            # 矩形の幅と高さから方向を判定
                                            rect_width = abs(p2[0] - p1[0])
                                            rect_height = abs(p2[1] - p1[1])
                                        
                                            if rect_width > rect_height:
                                                # X方向：x座標で最も離れた2本を選択
                                                walls_by_x = sorted(walls_in_selection, 
                                                                  key=lambda w: min(w['start'][0], w['end'][0]))
                                                leftmost_wall = walls_by_x[0]
                                                rightmost_wall = walls_by_x[-1]
                                            
                                                # 2本だけを結合候補として抽出
                                                selected_walls = [leftmost_wall, rightmost_wall]
                                                direction = "X方向"
                                            else:
                                                # Y方向：y座標で最も離れた2本を選択
                                                walls_by_y = sorted(walls_in_selection,
                                                                  key=lambda w: min(w['start'][1], w['end'][1]))
                                                bottom_wall = walls_by_y[0]
                                                top_wall = walls_by_y[-1]
                                            
                                                # 2本だけを結合候補として抽出
                                                selected_walls = [bottom_wall, top_wall]
                                                direction = "Y方向"
                                        
                                            st.write(f"**方向判定:** {direction} (幅: {rect_width}px, 高さ: {rect_height}px)")
                                            st.write(f"**結合対象:** 壁({selected_walls[0]['id']}) ← → 壁({selected_walls[1]['id']})")
                                        
                                            # 結合候補を探す（選択された2本だけ）
                                            candidates = _find_mergeable_walls(
                                                selected_walls,
                                                distance_threshold=distance_threshold,
                                                angle_threshold=15
                                            )
                                        
                                            if candidates:
                                                # 最有力候補の詳細情報を表示（デバッグ用）
                                                top_candidate = candidates[0]
                                                st.write(f"**検出されたペア：**")
                                                if top_candidate.get('is_chain', False):
                                                    chain_wall_ids = [w['id'] for w in top_candidate['walls']]
                                                    st.write(f"チェーン: {chain_wall_ids}")
                                                else:
                                                    st.write(f"ペア: 壁#{top_candidate['wall1']['id']} + 壁#{top_candidate['wall2']['id']}")
                                            
                                                # 最有力候補で結合
                                                updated_json = _merge_walls_in_json(updated_json, candidates[:1])
                                                total_merged_count += 1
                                            
                                                # 矩形内の他の不要な線分（中間線）を削除
                                                walls_to_delete = []
                                                for wall in walls_in_selection:
                                                    if wall['id'] not in [selected_walls[0]['id'], selected_walls[1]['id']]:
                                                        walls_to_delete.append(wall['id'])
                                            
                                                if walls_to_delete:
                                                    st.write(f"**削除対象の中間線:** 壁#{walls_to_delete}")
                                                    updated_json = _delete_walls_in_json(updated_json, walls_to_delete)
                                            
                                                color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                            
                                                # 結合詳細を記録
                                                merge_details.append({
                                                    'rect_idx': rect_idx,
                                                    'color_name': color_name,
                                                    'is_chain': False,
                                                    'walls': [selected_walls[0]['id'], selected_walls[1]['id']],
                                                    'distance': top_candidate['distance'],
                                                    'direction': direction,
                                                    'deleted_walls': walls_to_delete
                                                })
                                            else:
                                                st.warning(f"⚠️ 矩形内の壁が接続されていません")
                                
                                    if total_merged_count > 0:
                                        st.success(f"✅ 合計 {total_merged_count} 個の選択範囲で結合が完了しました")
                                    
                                        # 結合詳細を表示
                                        st.markdown("**結合結果:**")
                                        for detail in merge_details:
                                            result_text = (
                                                f"#{detail['rect_idx']+1}（{detail['color_name']}）: "
                                                f"壁({detail['walls'][0]}) ↔ 壁({detail['walls'][1]}) "
                                                f"({detail['direction']}) - "
                                                f"距離: {detail['distance']:.3f}m"
                                            )
                                            if detail.get('deleted_walls'):
                                                deleted_display = ", ".join([f"壁({wid})" for wid in detail['deleted_walls']])
                                                result_text += f" | 削除: {deleted_display}"
                                            st.write(result_text)
                                    else:
                                        st.warning("⚠️ 選択範囲内に結合可能な壁線が見つかりません")
                            
                                elif edit_mode == "線を追加":
                                    # ===== 線を追加モード =====
                                    total_added_count = 0
                                    add_details = []
                                
                                    for rect_idx, (p1, p2) in enumerate(target_rects):
                                        rect = {
                                            'left': min(p1[0], p2[0]),
                                            'top': min(p1[1], p2[1]),
                                            'width': abs(p2[0] - p1[0]),
                                            'height': abs(p2[1] - p1[1])
                                        }
                                    
                                        # 選択範囲内の壁線を抽出
                                        walls_in_selection = [
                                            wall for wall in updated_json['walls']
                                            if _wall_in_rect(wall, rect, scale, margin, img_height, min_x, min_y, max_x, max_y)
                                        ]
                                    
                                        # 最初の壁（wall1）の高さを取得、なければデフォルト高さを使用
                                        wall_height_to_use = None
                                        if len(walls_in_selection) > 0:
                                            wall_height_to_use = walls_in_selection[0].get('height', None)
                                    
                                        # 線を追加（スケールをセッション状態から取得）
                                        updated_json, direction, new_wall = _add_line_to_json(
                                            updated_json, p1, p2, wall_height=wall_height_to_use, scale=st.session_state.viz_scale
                                        )
                                        total_added_count += 1
                                    
                                        color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                        direction_jp = "縦線" if direction == "vertical" else "横線"
                                        add_details.append({
                                            'rect_idx': rect_idx,
                                            'color_name': color_name,
                                            'wall_id': new_wall['id'],
                                            'direction': direction_jp,
                                            'length': new_wall['length']
                                        })
                                
                                    if total_added_count > 0:
                                        st.success(f"✅ 合計 {total_added_count} 本の線を追加しました")
                                    
                                        # 追加詳細を表示
                                        st.markdown("**追加結果:**")
                                        for detail in add_details:
                                            st.write(
                                                f"#{detail['rect_idx']+1}（{detail['color_name']}）: "
                                                f"壁#{detail['wall_id']} - {detail['direction']} "
                                                f"（長さ: {detail['length']:.3f}m）"
                                            )
                            
                                elif edit_mode == "線を削除":
                                    # ===== 線を削除モード =====
                                    total_deleted_count = 0
                                    delete_details = []
                                    walls_to_delete = []  # 削除対象の壁IDリスト
                                
                                    for rect_idx, (p1, p2) in enumerate(target_rects):
                                        rect = {
                                            'left': min(p1[0], p2[0]),
                                            'top': min(p1[1], p2[1]),
                                            'width': abs(p2[0] - p1[0]),
                                            'height': abs(p2[1] - p1[1])
                                        }
                                    
                                        # 矩形内に完全に含まれる壁線を抽出
                                        walls_in_rect = _filter_walls_strictly_in_rect(
                                            updated_json['walls'], rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                        )
                                    
                                        if walls_in_rect:
                                            # 矩形内の壁をすべて削除対象に追加
                                            color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                            for wall in walls_in_rect:
                                                walls_to_delete.append(wall['id'])
                                                delete_details.append({
                                                    'rect_idx': rect_idx,
                                                    'color_name': color_name,
                                                    'wall_id': wall['id']
                                                })
                                
                                    if len(walls_to_delete) > 0:
                                        # 壁を削除
                                        updated_json = _delete_walls_in_json(updated_json, walls_to_delete)
                                        total_deleted_count = len(walls_to_delete)
                                    
                                        st.success(f"✅ 合計 {total_deleted_count} 本の壁を削除しました")
                                    
                                        # 削除詳細を表示
                                        st.markdown("**削除結果:**")
                                        for detail in delete_details:
                                            st.write(
                                                f"#{detail['rect_idx']+1}（{detail['color_name']}）: "
                                                f"壁({detail['wall_id']})を削除"
                                            )
                                    else:
                                        st.warning("⚠️ 削除対象の壁が見つかりません")
                            
                                # 一時ファイルに保存
                                temp_json_path = Path(st.session_state.out_dir) / "walls_3d_edited.json"
                                with open(temp_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(updated_json, f, indent=2, ensure_ascii=False)
                                
                                # 再可視化（元の変換と同じスケールを使用）
                                # 窓追加モードの場合は追加した壁を赤色で表示
                                temp_viz_path = Path(st.session_state.out_dir) / "visualization_edited.png"
                                highlight_ids = added_wall_ids if edit_mode == "窓を追加して結合" else None
                                visualize_3d_walls(str(temp_json_path), str(temp_viz_path), scale=int(viz_scale), highlight_wall_ids=highlight_ids)
                            
                                # 3Dビューア生成
                                temp_viewer_path = Path(st.session_state.out_dir) / "viewer_3d_edited.html"
                                _generate_3d_viewer_html(temp_json_path, temp_viewer_path)
                            
                                # セッション状態を更新（スケール校正で最新図を使用するため）
                                st.session_state.json_bytes = temp_json_path.read_bytes()
                                st.session_state.viz_bytes = temp_viz_path.read_bytes()
                            
                                # 編集後の画像を読み込み
                                edited_viz_bytes = temp_viz_path.read_bytes()
                                viewer_html_bytes = temp_viewer_path.read_bytes()
                            
                                # 編集結果をセッション状態に保存
                                if edit_mode == "線を結合":
                                    edit_count = total_merged_count
                                    edit_details = merge_details
                                elif edit_mode == "線を追加":
                                    edit_count = total_added_count
                                    edit_details = add_details
                                else:  # 線を削除
                                    edit_count = total_deleted_count
                                    edit_details = delete_details
                            
                                st.session_state.merge_result = {
                                    'original_viz_bytes': original_viz_bytes,
                                    'edited_viz_bytes': edited_viz_bytes,
                                    'json_data': original_json_data,
                                    'updated_json': updated_json,
                                    'temp_json_path': temp_json_path,
                                    'temp_viz_path': temp_viz_path,
                                    'temp_viewer_path': temp_viewer_path,
                                    'viewer_html_bytes': viewer_html_bytes,
                                    'edit_count': edit_count,
                                    'edit_details': edit_details
                                }
                                # 編集状態をリセット
                                st.session_state.rect_coords = []
                                st.session_state.rect_coords_list = []
                                # 窓追加パラメータもクリア
                                if 'window_execution_params' in st.session_state:
                                    del st.session_state.window_execution_params
                                st.rerun()
                        
                            except Exception as e:
                                st.error(f"エラーが発生しました: {e}")
                                import traceback
                                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
