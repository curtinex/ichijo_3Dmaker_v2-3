"""
Blenderで3D壁データをインポートするスクリプト
extract_walls_to_3d.pyで生成したJSONファイルを読み込んで、壁を3Dモデルとして生成します

使用方法:
1. Blenderを開く
2. Scripting タブを開く
3. このスクリプトを開く
4. json_path変数を生成したJSONファイルのパスに変更
5. スクリプトを実行（Alt+P または Run Script）
"""

import bpy
import json
import math
from pathlib import Path


def clear_scene():
    """シーン内のすべてのメッシュオブジェクトを削除"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_wall_from_line(wall_data: dict):
    """
    線分データから壁を作成
    
    wall_data形式:
    {
        "id": "wall_1",
        "start": [x1, y1, z1],
        "end": [x2, y2, z2],
        "height": 2.4,
        "thickness": 0.15
    }
    """
    start = wall_data['start']
    end = wall_data['end']
    height = wall_data['height']
    thickness = wall_data['thickness']
    
    # 壁の中心位置を計算
    center_x = (start[0] + end[0]) / 2
    center_y = (start[1] + end[1]) / 2
    center_z = height / 2  # 床から壁の中心まで
    
    # 壁の長さを計算
    length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    # 壁の回転角度を計算（Z軸周り）
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    
    # キューブを追加（壁として）
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(center_x, center_y, center_z)
    )
    
    wall = bpy.context.active_object
    wall.name = wall_data['id']
    
    # サイズを設定（X=長さ, Y=厚さ, Z=高さ）
    # JSONの値をそのまま使用（JSONで0.22と5.22を指定）
    wall.scale = (length, thickness, height)
    
    # 回転を設定（Z軸周り）
    wall.rotation_euler[2] = angle
    
    # マテリアルを設定
    mat_name = f"{wall_data['id']}_Material"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.diffuse_color = (0.9, 0.9, 0.9, 1.0)  # ライトグレー
    
    if wall.data.materials:
        wall.data.materials[0] = mat
    else:
        wall.data.materials.append(mat)
    
    return wall


def create_floor_plane(bounds, height=0.0, thickness=0.1):
    """
    床を作成
    
    bounds: {"min_x": ..., "max_x": ..., "min_y": ..., "max_y": ...}
    """
    center_x = (bounds['min_x'] + bounds['max_x']) / 2
    center_y = (bounds['min_y'] + bounds['max_y']) / 2
    size_x = bounds['max_x'] - bounds['min_x']
    size_y = bounds['max_y'] - bounds['min_y']
    
    # 床用の平面を追加
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(center_x, center_y, height - thickness/2)
    )
    
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (size_x, size_y, thickness)
    
    # マテリアルを設定
    mat = bpy.data.materials.new(name="Floor_Material")
    mat.diffuse_color = (0.7, 0.6, 0.5, 1.0)  # ベージュ
    floor.data.materials.append(mat)
    
    return floor


def import_walls_from_json(json_path, create_floor=True):
    """
    JSONファイルから壁データを読み込んで3Dモデルを生成
    """
    print(f"Loading: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    walls = data.get('walls', [])
    print(f"Total walls: {len(walls)}")
    
    if not walls:
        print("Error: No walls found in JSON")
        return
    
    # シーンをクリア
    clear_scene()
    
    # すべての壁を作成
    created_walls = []
    for wall_data in walls:
        wall = create_wall_from_line(wall_data)
        created_walls.append(wall)
    
    print(f"Created {len(created_walls)} walls")
    
    # 床を作成
    if create_floor:
        # 壁の範囲から床のサイズを計算
        all_x = []
        all_y = []
        for wall in walls:
            all_x.extend([wall['start'][0], wall['end'][0]])
            all_y.extend([wall['start'][1], wall['end'][1]])
        
        bounds = {
            'min_x': min(all_x) - 0.5,
            'max_x': max(all_x) + 0.5,
            'min_y': min(all_y) - 0.5,
            'max_y': max(all_y) + 0.5
        }
        
        floor = create_floor_plane(bounds)
        print("Created floor")
    
    # カメラとライトを追加
    setup_camera_and_light()
    
    print("Import complete!")


def setup_camera_and_light():
    """カメラとライトを設定"""
    # カメラを追加
    bpy.ops.object.camera_add(location=(15, -15, 15))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
    bpy.context.scene.camera = camera
    
    # ライトを追加
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    light = bpy.context.active_object
    light.data.energy = 2.0


# ========================================
# メイン実行部分
# ========================================

# ★ここにJSONファイルのパスを指定してください★
# v2推奨（最もバランスの良い結果: 25壁、85.39m）
# 更新版: thickness=0.22, height=5.22
json_path = r"C:\Users\curtin\Documents\DX\3_DMM生成AI\自由課題\Ichijo_v2\walls_3d_v2_updated.json"

# JSONが存在するか確認
if Path(json_path).exists():
    import_walls_from_json(json_path, create_floor=True)
else:
    print(f"Error: File not found: {json_path}")
    print("Please update the json_path variable with the correct path.")
