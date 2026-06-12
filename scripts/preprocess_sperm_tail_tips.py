"""预处理脚本：预先计算所有整精子图片的尾部尖端位置。

这个脚本会：
1. 扫描 data/pre-individual-obj/individual_obj/whole_sperm/ 中的所有图片
2. 对每张图片找到精子尾部尖端位置
3. 保存到 JSON 文件中

生成的 JSON 文件会被 SpermTailFromImageGenerator 加载使用。

用法：
    python scripts/preprocess_sperm_tail_tips.py [--output data/sperm_tail_tips.json]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def skeletonize_opencv(img: np.ndarray) -> np.ndarray:
    """使用形态学操作实现骨架化。"""
    img = img.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def find_tail_tip(mask: np.ndarray) -> tuple:
    """在 mask 中找到精子尾部尖端坐标。

    算法：
    1. 骨架化得到中心线
    2. 找骨架端点（只有1个邻居的像素）
    3. 端点中离质心最远的即为尾部尖端
    """
    # 骨架化
    skeleton = skeletonize_opencv(mask)

    # 找骨架端点
    h, w = skeleton.shape
    endpoints = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            neighbors = np.sum(skeleton[y - 1:y + 2, x - 1:x + 2] > 0) - 1
            if neighbors == 1:
                endpoints.append((x, y))

    if not endpoints:
        M = cv2.moments(mask)
        if M["m00"] > 0:
            return (M["m10"] / M["m00"], M["m01"] / M["m00"])
        return (mask.shape[1] / 2, mask.shape[0] / 2)

    # 计算质心
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = mask.shape[1] / 2
        cy = mask.shape[0] / 2

    # 端点中离质心最远的即为尾部尖端
    tail_tip = max(endpoints, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    return (float(tail_tip[0]), float(tail_tip[1]))


def process_image(img_path: Path) -> dict:
    """处理单张图片，返回结果字典。"""
    # 加载图片
    img_rgba = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_rgba is None:
        return {"file": img_path.name, "error": "无法加载图片"}

    # 提取 mask
    if img_rgba.ndim == 3 and img_rgba.shape[2] == 4:
        gray = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = img_rgba[:, :, 3]
        mask = (alpha > 128).astype(np.uint8) * 255
    elif img_rgba.ndim == 3:
        gray = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        gray = img_rgba
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学开运算清理噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 检查 mask 是否有效
    if np.count_nonzero(mask) == 0:
        return {"file": img_path.name, "error": "mask 为空"}

    # 找到尾部尖端
    tail_tip = find_tail_tip(mask)

    return {
        "file": img_path.name,
        "tail_tip": [tail_tip[0], tail_tip[1]],
        "image_size": [gray.shape[1], gray.shape[0]],  # (width, height)
        "mask_pixels": int(np.count_nonzero(mask)),
    }


def main():
    parser = argparse.ArgumentParser(description="预处理精子尾部尖端位置")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/pre-individual-obj/individual_obj/whole_sperm",
        help="整精子图片目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sperm_tail_tips.json",
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # 扫描图片
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
        image_paths.extend(sorted(input_dir.glob(ext)))

    print(f"找到 {len(image_paths)} 张图片")

    if not image_paths:
        print("没有找到图片，退出")
        return

    # 处理每张图片
    results = []
    errors = 0

    for i, img_path in enumerate(image_paths):
        result = process_image(img_path)
        results.append(result)

        if "error" in result:
            errors += 1
            print(f"[{i+1}/{len(image_paths)}] {img_path.name}: ERROR - {result['error']}")
        else:
            if (i + 1) % 100 == 0 or i == 0:
                tip = result["tail_tip"]
                print(f"[{i+1}/{len(image_paths)}] {img_path.name}: tail_tip=({tip[0]:.1f}, {tip[1]:.1f})")

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n完成！")
    print(f"  总图片数: {len(image_paths)}")
    print(f"  成功: {len(image_paths) - errors}")
    print(f"  失败: {errors}")
    print(f"  输出文件: {output_path}")


if __name__ == "__main__":
    main()
