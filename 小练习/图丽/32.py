import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class HighwayLaneDetector:
    def __init__(self, img_size=(720, 1280)):
        self.img_size = img_size
        self.white_lower = np.array([200, 200, 200])
        self.white_upper = np.array([255, 255, 255])

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enhance_image(self, image):
        """图像增强"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return bgr_enhanced

    def visualize_features(
        self, mask, stats, labels, features_list, components_indices, save_dir
    ):
        """在每个连通域上标注其特征值"""
        os.makedirs(save_dir, exist_ok=True)
        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 文字基本设置
        font_scale = 0.4
        font_thickness = 1
        line_spacing = 15  # 行间距
        x_margin = 10  # 左边距
        y_start = 20  # 顶部边距

        for feat_idx, comp_idx in enumerate(components_indices):
            features = features_list[feat_idx]
            _, _, _, _, area = stats[comp_idx]

            # 准备要显示的文本
            text_lines = [
                f"Region {comp_idx}:",
                f"Complexity: {features['shape_complexity']:.2f}",
                f"Ellipse Ratio: {features['ellipse_ratio']:.2f}",
                f"Vertical Deviation: {features['vertical_deviation']:.1f}",
                f"Convexity: {features['convexity']:.2f}",
                f"Rectangularity: {features['rectangularity']:.2f}",
                f"Area: {area}",
            ]

            # 为每个连通域创建单独的可视化图像
            component_mask = (labels == comp_idx).astype(np.uint8) * 255
            component_vis = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)

            # 在连通域图像上绘制特征文本
            y_pos = y_start
            for line in text_lines:
                # 计算文本大小
                (text_w, text_h), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )

                # 绘制黑色背景
                cv2.rectangle(
                    component_vis,
                    (x_margin, y_pos - text_h - 2),
                    (x_margin + text_w, y_pos + 2),
                    (0, 0, 0),
                    -1,
                )

                # 绘制白色文本
                cv2.putText(
                    component_vis,
                    line,
                    (x_margin, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                )
                y_pos += line_spacing

            # 保存单个连通域的可视化结果
            component_path = os.path.join(save_dir, f"component_{comp_idx}.png")
            cv2.imwrite(component_path, component_vis)

            # 在整体掩码图像上绘制文本
            y_pos = y_start + feat_idx * (len(text_lines) + 1) * line_spacing
            for line in text_lines:
                # 计算文本大小
                (text_w, text_h), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )

                # 绘制黑色背景
                cv2.rectangle(
                    vis_img,
                    (x_margin, y_pos - text_h - 2),
                    (x_margin + text_w, y_pos + 2),
                    (0, 0, 0),
                    -1,
                )

                # 绘制绿色文本
                cv2.putText(
                    vis_img,
                    line,
                    (x_margin, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                y_pos += line_spacing

        # 保存整体可视化结果
        cv2.imwrite(os.path.join(save_dir, "all_features.png"), vis_img)

    def detect_white_lines_with_visualization(self, image, save_dir):
        """带可视化的白线检测"""
        # 基础的颜色阈值分割和形态学处理
        mask = cv2.inRange(image, self.white_lower, self.white_upper)
        kernel = np.ones((3, 3), np.uint8)
        kernel_vertical = np.ones((5, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vertical)

        # 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filtered_mask = np.zeros_like(mask)
        height = mask.shape[0]
        min_area = max(height * 0.03, 50) * 2

        features_list = []
        valid_indices = []  # 存储有效的连通域索引

        for i in range(1, num_labels):
            area = stats[i][4]
            if area < min_area:
                continue

            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours or len(contours[0]) < 5:
                continue

            contour = contours[0]

            # 计算所有特征
            features = {}

            # 形状复杂度
            perimeter = cv2.arcLength(contour, True)
            features["shape_complexity"] = (4 * np.pi * area) / (perimeter * perimeter)

            # 椭圆特征
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]  # 椭圆方向原始值为(-90,90]
            # 将角度转换到[0,180)范围
            if angle < 0:
                angle += 180
            # 将角度映射到[0,90]范围
            if angle > 90:
                angle = 180 - angle
            features["vertical_deviation"] = abs(angle)

            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            features["ellipse_ratio"] = (
                major_axis / minor_axis if minor_axis > 0 else float("inf")
            )

            # 凸性
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            features["convexity"] = area / hull_area if hull_area > 0 else 0

            # 矩形度
            rect = cv2.minAreaRect(contour)
            rect_area = rect[1][0] * rect[1][1]
            features["rectangularity"] = area / rect_area if rect_area > 0 else 0

            features_list.append(features)
            valid_indices.append(i)  # 记录当前连通域的索引
            # 判断是否为线
            is_line = (
                features["shape_complexity"] < 0.5
                and features["ellipse_ratio"] > 4
                and features["vertical_deviation"] < 25
                and features["convexity"] > 0.4
                and features["rectangularity"] > 0.3
                and area > min_area
            )

            if is_line:
                filtered_mask = cv2.bitwise_or(
                    filtered_mask, component_mask.astype(np.uint8) * 255
                )

        # 可视化特征
        if features_list:
            self.visualize_features(
                mask, stats, labels, features_list, valid_indices, save_dir
            )

        # 最终的形态学处理
        filtered_mask = cv2.morphologyEx(
            filtered_mask, cv2.MORPH_CLOSE, kernel_vertical
        )
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)

        return filtered_mask, mask


def process_and_display(image_path, save_dir="output"):
    """处理图像并显示结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 初始化检测器
    detector = HighwayLaneDetector()

    # 处理图像
    image = detector.enhance_image(image)
    filtered_mask, original_mask = detector.detect_white_lines_with_visualization(
        image, save_dir
    )

    # 创建结果显示
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(original_mask, cmap="gray")
    plt.title("Original Mask")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(filtered_mask, cmap="gray")
    plt.title("Filtered Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"))
    plt.close()


def main():
    """主函数"""
    image_path = "simple.jpg"  # 替换为你的输入图像路径
    save_dir = "output"

    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 处理图像
    process_and_display(image_path, save_dir)
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
