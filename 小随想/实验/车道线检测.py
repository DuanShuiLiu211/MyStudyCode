import sys
import cv2
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cluster import DBSCAN


class HighwayLaneDetector:
    def __init__(self, img_size=(720, 1280)):
        self.img_size = img_size

        # 白线颜色阈值
        self.white_lower = np.array([200, 200, 200])
        self.white_upper = np.array([255, 255, 255])

        # 霍夫变换参数
        self.hough_params = {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 15,  # 降低阈值以检测更多线段
            "minLineLength": 40,  # 增加最小线段长度
            "maxLineGap": 50,  # 增加最大间隙以更好地连接线段
        }

        # 直方图均衡化
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enhance_image(self, image):
        """图像增强"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return bgr_enhanced

    def detect_white_lines(self, image):
        """白线检测 - 使用轮廓特征分析"""
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
        min_area = max(height * 0.05, 50) * 2  # 最小面积要求

        for i in range(1, num_labels):
            # 获取区域面积
            area = stats[i][4]
            if area < min_area:
                continue

            # 获取当前连通域的掩码和轮廓
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = contours[0]

            # 1. 计算形状复杂度（面积与周长的比）
            perimeter = cv2.arcLength(contour, True)
            shape_complexity = (4 * np.pi * area) / (perimeter * perimeter)

            # 2. 计算轮廓的拟合椭圆
            if len(contour) < 5:
                continue
            ellipse = cv2.fitEllipse(contour)

            # 获取椭圆的方向和扁率
            angle = ellipse[2]  # 椭圆方向原始值为(-90,90]
            # 将角度转换到[0,180)范围
            if angle < 0:
                angle += 180
            # 将角度映射到[0,90]范围
            if angle > 90:
                angle = 180 - angle
            vertical_deviation = abs(angle)

            # 计算椭圆的长短轴比（扁率）
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            ellipse_ratio = major_axis / minor_axis if minor_axis > 0 else float("inf")

            # 3. 计算轮廓的不凸度
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0

            # 4. 计算轮廓的矩形度
            rect = cv2.minAreaRect(contour)
            rect_area = rect[1][0] * rect[1][1]
            rectangularity = area / rect_area if rect_area > 0 else 0

            # 综合判断条件
            is_line = (
                shape_complexity < 0.5
                and ellipse_ratio > 4
                and vertical_deviation < 25
                and convexity > 0.4
                and rectangularity > 0.3
                and area > min_area
            )

            if is_line:
                filtered_mask = cv2.bitwise_or(
                    filtered_mask, component_mask.astype(np.uint8) * 255
                )

        # 最终的形态学处理
        filtered_mask = cv2.morphologyEx(
            filtered_mask, cv2.MORPH_CLOSE, kernel_vertical
        )
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)

        return filtered_mask

    def detect_lines(self, binary_image):
        """线段检测 - 改进的霍夫变换和线段合并"""
        # 进行形态学操作以增强垂直线条
        kernel_vertical = np.ones((3, 1), np.uint8)
        enhanced = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_vertical)

        # 霍夫变换检测线段
        lines = cv2.HoughLinesP(
            enhanced,
            self.hough_params["rho"],
            self.hough_params["theta"],
            self.hough_params["threshold"],
            minLineLength=self.hough_params["minLineLength"],
            maxLineGap=self.hough_params["maxLineGap"],
        )

        if lines is None:
            return []

        # 预处理检测到的线段
        height = binary_image.shape[0]
        preliminary_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 确保y1总是较小的y值
            if y2 < y1:
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            # 计算角度和长度
            dx = x2 - x1
            dy = y2 - y1
            angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
            length = np.sqrt(dx * dx + dy * dy)

            # 使用更宽松的角度条件
            if (60 < angle < 120) and (length > height * 0.05):
                preliminary_lines.append([(x1, y1), (x2, y2), angle, length])

        if not preliminary_lines:
            return []

        # 按x坐标分组
        x_tolerance = binary_image.shape[1] * 0.02
        preliminary_lines.sort(key=lambda l: (l[0][0] + l[1][0]) / 2)  # 按平均x坐标排序

        merged_lines = []
        current_group = [preliminary_lines[0]]

        for line in preliminary_lines[1:]:
            prev_line = current_group[-1]
            curr_x = (line[0][0] + line[1][0]) / 2
            prev_x = (prev_line[0][0] + prev_line[1][0]) / 2

            # 检查是否属于同一组
            if abs(curr_x - prev_x) < x_tolerance:
                # 检查垂直重叠
                y_overlap = min(line[1][1], prev_line[1][1]) - max(
                    line[0][1], prev_line[0][1]
                )

                if y_overlap > -self.hough_params["maxLineGap"]:
                    current_group.append(line)
                else:
                    # 合并当前组并开始新组
                    if len(current_group) > 0:
                        merged_line = self._merge_line_group(current_group)
                        if merged_line is not None:
                            merged_lines.append(merged_line)
                    current_group = [line]
            else:
                # 合并当前组并开始新组
                if len(current_group) > 0:
                    merged_line = self._merge_line_group(current_group)
                    if merged_line is not None:
                        merged_lines.append(merged_line)
                current_group = [line]

        # 处理最后一组
        if len(current_group) > 0:
            merged_line = self._merge_line_group(current_group)
            if merged_line is not None:
                merged_lines.append(merged_line)

        # 转换回原始格式 [x1, y1, x2, y2]
        return [
            [int(x1), int(y1), int(x2), int(y2)]
            for (x1, y1), (x2, y2), _, _ in merged_lines
        ]

    def _merge_line_group(self, lines):
        """合并一组线段"""
        if not lines:
            return None

        # 如果只有一条线段
        if len(lines) == 1:
            return lines[0]

        # 获取所有端点
        points = []
        for line in lines:
            points.append(line[0])  # 起点
            points.append(line[1])  # 终点

        # 按y坐标排序
        points.sort(key=lambda p: p[1])

        # 获取最高和最低的点
        top_point = points[0]
        bottom_point = points[-1]

        # 计算平均角度
        avg_angle = np.mean([line[2] for line in lines])

        # 计算合并后的长度
        length = np.sqrt(
            (bottom_point[0] - top_point[0]) ** 2
            + (bottom_point[1] - top_point[1]) ** 2
        )

        return [top_point, bottom_point, avg_angle, length]

    def classify_lane_markings(self, lines, binary_image):
        """基于曲线拟合的车道线分类"""
        if not lines:
            return [], []

        height, width = binary_image.shape
        x_tolerance = width * 0.02

        # 按x坐标分组
        lines_x = [(line, (line[0] + line[2]) / 2) for line in lines]
        lines_x.sort(key=lambda x: x[1])  # 按x坐标排序

        # 使用scikit-learn的DBSCAN聚类分组
        x_coords = np.array([x for _, x in lines_x])
        x_coords = x_coords.reshape(-1, 1)
        db = DBSCAN(eps=x_tolerance * 3, min_samples=2)
        labels = db.fit_predict(x_coords)

        # 创建分组
        groups = {}
        for i, label in enumerate(labels):
            if label < 0:  # 噪声点
                continue
            if label not in groups:
                groups[label] = []
            groups[label].append(lines_x[i][0])

        solid_lines = []
        dashed_lines = []

        for group in groups.values():
            # 如果组内只有一条线
            if len(group) == 1:
                line = group[0]
                length = np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
                if length > height * 0.25:
                    solid_lines.append(line)
                else:
                    dashed_lines.append(line)
                continue

            # 收集所有线段的端点
            points = []
            for line in group:
                points.append([line[0], line[1]])  # 起点
                points.append([line[2], line[3]])  # 终点
            points = np.array(points)

            # 按y坐标排序
            points = points[points[:, 1].argsort()]

            # 尝试拟合多项式
            try:
                # 使用2阶多项式拟合
                coeffs = np.polyfit(points[:, 1], points[:, 0], 2)
                poly = np.poly1d(coeffs)

                # 计算拟合误差
                fitted_x = poly(points[:, 1])
                errors = np.abs(fitted_x - points[:, 0])
                avg_error = np.mean(errors)

                # 如果拟合误差较小，说明这些点大致在同一条曲线上
                if avg_error < x_tolerance * 2:
                    # 计算首尾两点间的距离
                    total_length = np.sqrt(
                        (points[-1][0] - points[0][0]) ** 2
                        + (points[-1][1] - points[0][1]) ** 2
                    )

                    # 计算所有线段的平均长度
                    avg_seg_length = np.mean(
                        [
                            np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
                            for line in group
                        ]
                    )

                    # 如果总长度大但平均线段长度小，归类为虚线
                    if total_length > height * 0.25 and avg_seg_length < height * 0.15:
                        dashed_lines.extend(group)
                    else:
                        for line in group:
                            length = np.sqrt(
                                (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2
                            )
                            if length > height * 0.25:
                                solid_lines.append(line)
                            else:
                                dashed_lines.append(line)
                else:
                    # 拟合误差大，按单条线段处理
                    for line in group:
                        length = np.sqrt(
                            (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2
                        )
                        if length > height * 0.25:
                            solid_lines.append(line)
                        else:
                            dashed_lines.append(line)

            except np.linalg.LinAlgError:
                # 如果拟合失败，按单条线段处理
                for line in group:
                    length = np.sqrt(
                        (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2
                    )
                    if length > height * 0.25:
                        solid_lines.append(line)
                    else:
                        dashed_lines.append(line)

        return solid_lines, dashed_lines

    def draw_result(self, image, solid_lines, dashed_lines):
        """在图像上绘制检测结果"""
        result = image.copy()

        # 绘制实线（红色）
        for line in solid_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 绘制虚线（绿色）
        for line in dashed_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result

    def process_image(self, image, debug=False):
        """处理单帧图像"""
        # 调整图像大小
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))

        # 图像增强
        enhanced = self.enhance_image(image)

        # 颜色检测
        white_mask = self.detect_white_lines(enhanced)

        # 检测线段
        lines = self.detect_lines(white_mask)

        # 创建霍夫变换的可视化结果
        hough_result = image.copy()
        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(hough_result, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 分类车道线
        solid_lines, dashed_lines = self.classify_lane_markings(lines, white_mask)

        # 绘制最终结果
        result = self.draw_result(image, solid_lines, dashed_lines)

        if debug:
            return result, white_mask, hough_result
        return result


def process_video(input_path, output_path):
    """处理视频文件，将车道线检测结果写入新视频"""
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # 获取视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # 或使用 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 创建车道线检测器
    detector = HighwayLaneDetector(img_size=(height, width))

    # 进度条所需变量
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理当前帧
            result = np.array(detector.process_image(frame), dtype=np.uint8)
            out.write(result)

            # 更新进度
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing: {progress:.1f}%", end="")

    finally:
        # 清理资源
        cap.release()
        out.release()
        print("\nVideo processing completed")


def main():
    """测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Highway Lane Detection")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video"],
        default="image",
        help="Processing mode: image or video",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input image/video file path"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path (optional)"
    )

    args = parser.parse_args()

    if args.mode == "image":
        # 读取测试图像
        image = cv2.imread(args.input)
        if image is None:
            print("Error: Could not read the image")
            return

        # 初始化检测器
        detector = HighwayLaneDetector()

        # 处理图像
        result, white_mask, hough_result = detector.process_image(image, debug=True)

        # 显示结果
        plot.figure(figsize=(15, 10))

        plot.subplot(221)
        plot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot.title("Original Image")
        plot.axis("off")

        plot.subplot(222)
        plot.imshow(white_mask, cmap="gray")
        plot.title("White Line Mask")
        plot.axis("off")

        plot.subplot(223)
        plot.imshow(cv2.cvtColor(hough_result, cv2.COLOR_BGR2RGB))
        plot.title("Hough Transform Result")
        plot.axis("off")

        plot.subplot(224)
        plot.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plot.title("Final Result\nSolid(Red) & Dashed(Green)")
        plot.axis("off")

        plot.tight_layout()

        if args.output:
            plot.savefig(args.output)
        plot.show()

    elif args.mode == "video":
        output_path = args.output if args.output else "output.mp4"
        process_video(args.input, output_path)


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--mode",
            "video",
            "--input",
            "./datas/videos/1.mp4",
        ]
    )
    main()
