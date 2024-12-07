import cv2
import numpy as np
import matplotlib.pyplot as plot


class HighwayLaneDetector:
    def __init__(self, img_size=(720, 1280)):
        self.img_size = img_size

        self.white_lower = np.array([200, 200, 200])
        self.white_upper = np.array([255, 255, 255])

        self.edge_params = {
            "low_threshold": 50,
            "high_threshold": 150,
            "aperture_size": 3,
        }

        # 霍夫变换参数
        self.hough_params = {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 20,
            "minLineLength": 30,
            "maxLineGap": 35,
        }

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enhance_image(self, image):
        """图像增强"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_white_lines(self, image):
        """白线检测"""
        mask = cv2.inRange(image, self.white_lower, self.white_upper)

        # 轻微的形态学处理来减少噪声
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask

    def detect_edges(self, image):
        """边缘检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用更小的核进行模糊,保留更多细节
        kernel_size = 3
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Canny边缘检测
        edges = cv2.Canny(
            blurred,
            self.edge_params["low_threshold"],
            self.edge_params["high_threshold"],
            apertureSize=self.edge_params["aperture_size"],
        )

        return edges

    def combine_detections(self, white_mask, edges):
        """智能组合白线检测和边缘检测的特征"""
        height, width = white_mask.shape

        # 1. 分别增强两种特征中的垂直线条
        kernel_vertical = np.ones((5, 1), np.uint8)
        white_vertical = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_vertical)
        edges_vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vertical)

        # 2. 对白线掩码进行连通域分析，分别处理实线和虚线特征
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            white_vertical, connectivity=8
        )
        white_filtered = np.zeros_like(white_mask)

        # 虚线特征参数
        min_height_dashed = height * 0.03  # 降低虚线最小高度要求
        max_width_dashed = width * 0.03  # 适当降低宽度限制

        # 实线特征参数
        min_height_solid = height * 0.1  # 实线需要更长
        max_width_solid = width * 0.05  # 实线可以稍宽

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            aspect_ratio = h / w if w > 0 else float("inf")

            # 分别处理实线和虚线特征
            is_potential_solid = (
                h > min_height_solid and w < max_width_solid and aspect_ratio > 3
            )
            is_potential_dashed = (
                h > min_height_dashed
                and h < min_height_solid
                and w < max_width_dashed
                and aspect_ratio > 1.5
            )

            if is_potential_solid or is_potential_dashed:
                component_mask = (labels == i).astype(np.uint8) * 255
                white_filtered = cv2.bitwise_or(white_filtered, component_mask)

        # 3. 对边缘检测结果进行类似处理
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            edges_vertical, connectivity=8
        )
        edges_filtered = np.zeros_like(edges)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            aspect_ratio = h / w if w > 0 else float("inf")

            # 使用相同的实虚线判断标准
            is_potential_solid = (
                h > min_height_solid and w < max_width_solid and aspect_ratio > 3
            )
            is_potential_dashed = (
                h > min_height_dashed
                and h < min_height_solid
                and w < max_width_dashed
                and aspect_ratio > 1.5
            )

            if is_potential_solid or is_potential_dashed:
                component_mask = (labels == i).astype(np.uint8) * 255
                edges_filtered = cv2.bitwise_or(edges_filtered, component_mask)

        # 4. 智能组合两种过滤后的特征
        # 首先找到两种特征都检测到的区域
        overlap = cv2.bitwise_and(white_filtered, edges_filtered)

        # 然后添加白线检测中置信度高的区域
        high_conf_white = cv2.dilate(overlap, np.ones((3, 3), np.uint8))
        white_confident = cv2.bitwise_and(
            white_filtered, cv2.bitwise_not(high_conf_white)
        )

        # 最后添加边缘检测中与现有特征接近的区域
        combined = cv2.bitwise_or(overlap, white_confident)
        edge_complement = cv2.bitwise_and(edges_filtered, cv2.bitwise_not(combined))

        # 进行最后的形态学清理
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        return combined

    def detect_lines(self, binary_image):
        """改进线段检测的过滤逻辑"""
        lines = cv2.HoughLinesP(
            binary_image,
            self.hough_params["rho"],
            self.hough_params["theta"],
            self.hough_params["threshold"],
            minLineLength=self.hough_params["minLineLength"],
            maxLineGap=self.hough_params["maxLineGap"],
        )

        if lines is None:
            return []

        filtered_lines = []
        height = binary_image.shape[0]

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算角度
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # 计算长度
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 更宽松的角度条件,但仍然保持合理范围
            if (70 < angle < 110) and (length > height * 0.1):
                filtered_lines.append(line[0])

        return filtered_lines

    def classify_lane_markings(self, lines, binary_image):
        """改进车道线分类方法"""
        if not lines:
            return [], []

        height, width = binary_image.shape
        x_tolerance = width * 0.02

        # 按x坐标分组
        lines_x = [(line, (line[0] + line[2]) / 2) for line in lines]

        # 特别处理最左和最右的线段组
        left_edge = width
        right_edge = 0
        for _, x in lines_x:
            left_edge = min(left_edge, x)
            right_edge = max(right_edge, x)

        # 分组
        lane_groups = {}
        for line, x in lines_x:
            group_id = int(x // x_tolerance)
            if group_id not in lane_groups:
                lane_groups[group_id] = []
            lane_groups[group_id].append(line)

        solid_lines = []
        dashed_lines = []

        for group_id, group in lane_groups.items():
            # 计算这组线段的平均x坐标
            avg_x = sum((line[0] + line[2]) / 2 for line in group) / len(group)

            # 按y坐标排序
            group.sort(key=lambda l: min(l[1], l[3]))

            # 分析垂直连续性
            y_segments = []
            current_segment = [group[0]]
            max_gap = height * 0.1  # 适当增加允许的间隙

            for i in range(1, len(group)):
                prev_line = current_segment[-1]
                curr_line = group[i]

                prev_bottom = max(prev_line[1], prev_line[3])
                curr_top = min(curr_line[1], curr_line[3])
                gap = curr_top - prev_bottom

                if gap < max_gap:
                    current_segment.append(curr_line)
                else:
                    y_segments.append(current_segment)
                    current_segment = [curr_line]

            y_segments.append(current_segment)

            # 分析特征
            total_length = sum(
                np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
                for segment in y_segments
                for line in segment
            )

            coverage = total_length / height

            # 根据位置和覆盖率判断
            is_edge_line = (abs(avg_x - left_edge) < x_tolerance * 2) or (
                abs(avg_x - right_edge) < x_tolerance * 2
            )

            if (is_edge_line and coverage > 0.5) or (coverage > 0.7):
                solid_lines.extend([line for segment in y_segments for line in segment])
            else:
                dashed_lines.extend(
                    [line for segment in y_segments for line in segment]
                )

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

        # 边缘检测
        edges = self.detect_edges(enhanced)

        # 组合特征
        combined = self.combine_detections(white_mask, edges)

        # 检测线段
        lines = self.detect_lines(combined)

        # 创建霍夫变换的可视化结果
        hough_result = image.copy()
        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(hough_result, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 分类车道线
        solid_lines, dashed_lines = self.classify_lane_markings(lines, combined)

        # 绘制最终结果
        result = self.draw_result(image, solid_lines, dashed_lines)

        if debug:
            return result, white_mask, edges, combined, hough_result
        return result


def main():
    """测试函数"""
    # 初始化检测器
    detector = HighwayLaneDetector()

    # 读取测试图像
    image = cv2.imread("sample.jpg")
    if image is None:
        print("Error: Could not read the image")
        return

    # 处理图像
    result, white_mask, edges, combined, hough_result = detector.process_image(
        image, debug=True
    )

    # 显示结果
    plot.figure(figsize=(20, 10))

    plot.subplot(231)
    plot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plot.title("Original Image")
    plot.axis("off")

    plot.subplot(232)
    plot.imshow(white_mask, cmap="gray")
    plot.title("White Line Mask")
    plot.axis("off")

    plot.subplot(233)
    plot.imshow(edges, cmap="gray")
    plot.title("Edge Detection")
    plot.axis("off")

    plot.subplot(234)
    plot.imshow(combined, cmap="gray")
    plot.title("Combined Detection")
    plot.axis("off")

    plot.subplot(235)
    plot.imshow(cv2.cvtColor(hough_result, cv2.COLOR_BGR2RGB))
    plot.title("Hough Transform Result")
    plot.axis("off")

    plot.subplot(236)
    plot.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plot.title("Final Result\nSolid(Red) & Dashed(Green)")
    plot.axis("off")

    plot.tight_layout()
    plot.show()


if __name__ == "__main__":
    main()
