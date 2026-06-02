"""
精灵检测模块

两阶段检测:
1. YOLO 目标检测 — 精灵定位 (RKNN NPU 优先, ONNX CPU fallback)
2. 模板匹配 — UI 元素检测 (丢球按钮、确认框等)

输出统一的检测结果列表: [{"bbox": (x1,y1,x2,y2), "class": str, "confidence": float}, ...]
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import log

# ============================================================
# 常量
# ============================================================

# COCO 风格类别 (如果使用通用 YOLO 模型，需替换为精灵类别)
DEFAULT_CLASSES = ["normal", "shiny", "corrupted"]

# 模板匹配默认模板名
UI_TEMPLATES = {
    "throw_button": "throw_button.png",
    "confirm_button": "confirm_button.png",
    "battle_end": "battle_end.png",
    "ball_icon": "ball_icon.png",
}


# ============================================================
# YOLO 后处理
# ============================================================

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """非极大值抑制，返回保留的索引列表。"""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def yolo_postprocess(output: np.ndarray,
                      input_shape: Tuple[int, int],
                      frame_shape: Tuple[int, int],
                      conf_threshold: float = 0.5,
                      nms_threshold: float = 0.4,
                      num_classes: int = 3) -> List[dict]:
    """
    YOLO 输出后处理。

    支持两种常见 YOLO 输出格式:
    - [1, N, 4+1+C] (含 objectness)
    - [1, N, 4+C]   (不含 objectness，只有 class scores)

    Args:
        output: 模型原始输出
        input_shape: 模型输入 (w, h)，如 (640, 640)
        frame_shape: 原始帧 (h, w, c)
        conf_threshold: 置信度阈值
        nms_threshold: NMS IoU 阈值
        num_classes: 类别数

    Returns:
        detections: [{"bbox": (x1,y1,x2,y2), "class": str, "confidence": float}, ...]
    """
    # 展平
    if output.ndim == 3:
        output = output.squeeze(0)  # (1, N, D) → (N, D)

    if output.size == 0 or output.ndim != 2:
        return []

    num_dims = output.shape[1]

    # 判断格式
    if num_dims == 4 + num_classes:
        # 无 objectness: [cx, cy, w, h, cls_0, ..., cls_C]
        has_objectness = False
    elif num_dims == 5 + num_classes:
        # 有 objectness: [cx, cy, w, h, obj, cls_0, ..., cls_C]
        has_objectness = True
    else:
        log.warning(f"Unexpected YOLO output dims: {output.shape}, expected "
                    f"{4 + num_classes} or {5 + num_classes}")
        # 尝试兼容: 假设前4个是bbox
        if num_dims < 5:
            return []
        num_classes = num_dims - 4
        has_objectness = (num_dims == 5 + num_classes)
        if num_dims == 4 + num_classes:
            has_objectness = False

    # 解析
    boxes_raw = output[:, :4].copy()  # cx, cy, w, h (归一化 0-1)

    if has_objectness:
        obj_conf = output[:, 4:5]
        class_scores = output[:, 5:5 + num_classes]
        scores = obj_conf.squeeze(-1) * class_scores.max(axis=1)
    else:
        class_scores = output[:, 4:4 + num_classes]
        scores = class_scores.max(axis=1)

    class_ids = class_scores.argmax(axis=1)

    # 置信度过滤
    mask = scores >= conf_threshold
    boxes_raw = boxes_raw[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_raw) == 0:
        return []

    # 坐标转换: cx,cy,w,h (归一化) → x1,y1,x2,y2 (像素坐标)
    in_w, in_h = input_shape
    f_h, f_w = frame_shape[:2]
    scale_x = f_w / in_w
    scale_y = f_h / in_h

    px_boxes = np.zeros_like(boxes_raw)
    px_boxes[:, 0] = (boxes_raw[:, 0] - boxes_raw[:, 2] / 2) * scale_x  # x1
    px_boxes[:, 1] = (boxes_raw[:, 1] - boxes_raw[:, 3] / 2) * scale_y  # y1
    px_boxes[:, 2] = (boxes_raw[:, 0] + boxes_raw[:, 2] / 2) * scale_x  # x2
    px_boxes[:, 3] = (boxes_raw[:, 1] + boxes_raw[:, 3] / 2) * scale_y  # y2

    # NMS
    keep = _nms(px_boxes, scores, nms_threshold)

    # 构建结果
    results = []
    for idx in keep:
        x1, y1, x2, y2 = px_boxes[idx].astype(int).tolist()
        cls_id = int(class_ids[idx])
        results.append({
            "bbox": (max(0, x1), max(0, y1),
                     min(f_w, x2), min(f_h, y2)),
            "class": DEFAULT_CLASSES[cls_id] if cls_id < len(DEFAULT_CLASSES) else f"cls_{cls_id}",
            "confidence": float(scores[idx]),
        })

    return results


# ============================================================
# 模板匹配引擎 (UI 检测)
# ============================================================

class TemplateMatcher:
    """基于 OpenCV 模板匹配的 UI 元素检测器。"""

    def __init__(self, templates_dir: str = "templates/",
                 threshold: float = 0.7):
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self._templates: dict = {}  # name → (image, w, h)
        self._load_templates()

    def _load_templates(self):
        """加载模板目录下所有 PNG 图片。"""
        if not self.templates_dir.exists():
            log.warning(f"Templates directory not found: {self.templates_dir}")
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            return

        for tmpl_path in self.templates_dir.glob("*.png"):
            name = tmpl_path.stem
            img = cv2.imread(str(tmpl_path), cv2.IMREAD_COLOR)
            if img is not None:
                self._templates[name] = {
                    "image": img,
                    "w": img.shape[1],
                    "h": img.shape[0],
                }
                log.debug(f"Loaded template: {name} ({img.shape[1]}x{img.shape[0]})")

        if self._templates:
            log.info(f"Loaded {len(self._templates)} templates: "
                     f"{list(self._templates.keys())}")
        else:
            log.info("No templates loaded. Add PNG files to templates/ dir.")

    def reload(self):
        """重新加载模板 (热更新)。"""
        self._templates.clear()
        self._load_templates()

    def detect(self, frame: np.ndarray,
               names: Optional[List[str]] = None) -> List[dict]:
        """
        在帧中检测指定模板。

        Args:
            frame: BGR 图像
            names: 要检测的模板名列表 (None = 全部)

        Returns:
            [{"name": str, "bbox": (x1,y1,x2,y2), "confidence": float}, ...]
        """
        if names is None:
            names = list(self._templates.keys())

        results = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for name in names:
            tmpl_info = self._templates.get(name)
            if tmpl_info is None:
                continue

            tmpl_img = tmpl_info["image"]
            gray_tmpl = cv2.cvtColor(tmpl_img, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= self.threshold:
                x, y = max_loc
                w, h = tmpl_info["w"], tmpl_info["h"]
                results.append({
                    "name": name,
                    "bbox": (x, y, x + w, y + h),
                    "confidence": float(max_val),
                })

        return results

    @property
    def template_names(self) -> List[str]:
        return list(self._templates.keys())


# ============================================================
# YOLO 检测器: RKNN 版本
# ============================================================

class RKNNDetector:
    """使用 RKNN Runtime (NPU) 进行 YOLO 推理。"""

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._rknn = None

    def load(self) -> bool:
        """加载 RKNN 模型到 NPU。"""
        if not os.path.exists(self.model_path):
            log.error(f"RKNN model not found: {self.model_path}")
            return False

        try:
            from rknnlite.api import RKNNLite
            self._rknn = RKNNLite()
            ret = self._rknn.load_rknn(self.model_path)
            if ret != 0:
                log.error(f"RKNN load failed: {ret}")
                return False

            ret = self._rknn.init_runtime(
                core_mask=RKNNLite.NPU_CORE_AUTO
            )
            if ret != 0:
                log.error(f"RKNN init_runtime failed: {ret}")
                return False

            log.info(f"RKNN model loaded: {self.model_path}")
            return True

        except ImportError:
            log.error("rknnlite not installed")
            return False
        except Exception as e:
            log.error(f"RKNN load error: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        对单帧运行 YOLO 检测。

        Returns:
            [{"bbox": (x1,y1,x2,y2), "class": str, "confidence": float}, ...]
        """
        if self._rknn is None:
            return []

        in_w, in_h = self.input_size

        # 预处理
        input_data, _ = self._preprocess(frame, (in_w, in_h))

        # 推理
        try:
            outputs = self._rknn.inference(inputs=[input_data])
        except Exception as e:
            log.error(f"RKNN inference error: {e}")
            return []

        if not outputs:
            return []

        # 后处理
        output = outputs[0]
        return yolo_postprocess(
            output,
            input_shape=(in_w, in_h),
            frame_shape=frame.shape,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )

    def _preprocess(self, frame: np.ndarray,
                    target_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float]]:
        """预处理: resize + normalize + channel transpose."""
        in_w, in_h = target_size
        f_h, f_w = frame.shape[:2]

        # Letterbox resize (保持宽高比, 填充灰边)
        scale = min(in_w / f_w, in_h / f_h)
        new_w, new_h = int(f_w * scale), int(f_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # 创建 letterbox 画布 (BGR, 填充 114)
        letterbox = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        dx = (in_w - new_w) // 2
        dy = (in_h - new_h) // 2
        letterbox[dy:dy + new_h, dx:dx + new_w] = resized

        # BGR → RGB, HWC → CHW, uint8 → float32, /255
        blob = letterbox[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # (1, 3, H, W)

        return blob, (scale, dx, dy)

    def release(self):
        if self._rknn:
            self._rknn.release()
            self._rknn = None

    def __del__(self):
        self.release()


# ============================================================
# YOLO 检测器: ONNX 版本 (CPU fallback)
# ============================================================

class ONNXDetector:
    """使用 ONNX Runtime (CPU) 进行 YOLO 推理。"""

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._session = None
        self._input_name = None
        self._output_names = None

    def load(self) -> bool:
        """加载 ONNX 模型。"""
        if not os.path.exists(self.model_path):
            log.error(f"ONNX model not found: {self.model_path}")
            return False

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 2

            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]

            log.info(f"ONNX model loaded: {self.model_path}")
            return True

        except ImportError:
            log.error("onnxruntime not installed")
            return False
        except Exception as e:
            log.error(f"ONNX load error: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[dict]:
        """对单帧运行 YOLO 检测。"""
        if self._session is None:
            return []

        in_w, in_h = self.input_size

        # 预处理
        input_data, _ = self._preprocess(frame, (in_w, in_h))

        # 推理
        try:
            outputs = self._session.run(
                self._output_names,
                {self._input_name: input_data}
            )
        except Exception as e:
            log.error(f"ONNX inference error: {e}")
            return []

        if not outputs:
            return []

        # 后处理 — 使用第一个输出
        output = outputs[0]
        return yolo_postprocess(
            output,
            input_shape=(in_w, in_h),
            frame_shape=frame.shape,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )

    def _preprocess(self, frame: np.ndarray,
                    target_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float]]:
        """同 RKNN 预处理。"""
        in_w, in_h = target_size
        f_h, f_w = frame.shape[:2]

        scale = min(in_w / f_w, in_h / f_h)
        new_w, new_h = int(f_w * scale), int(f_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        letterbox = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        dx = (in_w - new_w) // 2
        dy = (in_h - new_h) // 2
        letterbox[dy:dy + new_h, dx:dx + new_w] = resized

        blob = letterbox[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, (scale, dx, dy)

    def release(self):
        if self._session:
            del self._session
            self._session = None


# ============================================================
# 开放世界: 基于运动检测 + 颜色检测的精灵发现器
# (无需预训练模型，利用游戏特性)
# ============================================================

class MotionSpriteDetector:
    """
    无模型精灵发现器。

    利用游戏特性:
    - 精灵在场景中移动 (运动检测)
    - 精灵通常有鲜明颜色 (颜色显著性)
    - 精灵上方可能有名字标签 (文字区域)

    作为 YOLO 模型的备用方案。
    """

    def __init__(self,
                 motion_threshold: int = 500,
                 min_contour_area: int = 800):
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self._prev_frame: Optional[np.ndarray] = None
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40, detectShadows=False
        )

    def detect(self, frame: np.ndarray) -> List[dict]:
        """检测画面中可能包含精灵的区域。"""
        results = []

        if self._prev_frame is None:
            self._bg_subtractor.apply(frame)
            self._prev_frame = frame.copy()
            return results

        # 方法1: 背景减除 (检测运动物体)
        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # 方法2: 差值检测
        if self._prev_frame is not None:
            diff = cv2.absdiff(frame, self._prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_or(fg_mask, thresh)

        self._prev_frame = frame.copy()

        # 膨胀 + 寻找轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            results.append({
                "bbox": (x, y, x + w, y + h),
                "class": "motion_candidate",
                "confidence": min(1.0, area / 5000),
            })

        # 按面积排序，限制数量
        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results[:10]


# ============================================================
# 统一检测器 (策略模式)
# ============================================================

class SpriteDetector:
    """
    统一精灵检测器。

    自动选择最佳可用后端:
    1. RKNN YOLO (NPU) — 最佳性能
    2. ONNX YOLO (CPU) — 备选
    3. 运动检测 — 无模型应急方案

    所有后端 + 模板匹配同时运行。
    """

    def __init__(self, config: dict):
        """
        Args:
            config: detector 部分的配置字典
        """
        self.config = config

        # YOLO (RKNN)
        self._rknn: Optional[RKNNDetector] = None
        rknn_path = config.get("rknn_model", "")
        if rknn_path and os.path.exists(rknn_path):
            self._rknn = RKNNDetector(
                model_path=rknn_path,
                input_size=tuple(config.get("input_size", [640, 640])),
                conf_threshold=config.get("conf_threshold", 0.5),
                nms_threshold=config.get("nms_threshold", 0.4),
            )
            log.info(f"Attempting to load RKNN model: {rknn_path}")
            if not self._rknn.load():
                log.warning("RKNN load failed, will fall back")
                self._rknn = None

        # YOLO (ONNX fallback)
        self._onnx: Optional[ONNXDetector] = None
        if self._rknn is None:
            onnx_path = config.get("onnx_model", "")
            if onnx_path and os.path.exists(onnx_path):
                self._onnx = ONNXDetector(
                    model_path=onnx_path,
                    input_size=tuple(config.get("input_size", [640, 640])),
                    conf_threshold=config.get("conf_threshold", 0.5),
                    nms_threshold=config.get("nms_threshold", 0.4),
                )
                if not self._onnx.load():
                    log.warning("ONNX load failed, will use motion detection")
                    self._onnx = None

        # 运动检测 (最后兜底)
        self._motion: Optional[MotionSpriteDetector] = None
        if self._rknn is None and self._onnx is None:
            log.info("No YOLO model available, using motion-based detection")
            self._motion = MotionSpriteDetector()

        # 模板匹配
        tc = config.get("template", {})
        self._templates = TemplateMatcher(
            templates_dir=tc.get("templates_dir", "templates/"),
            threshold=tc.get("match_threshold", 0.7),
        )

        # 统计
        self._inference_time = 0.0
        self._detect_count = 0

    # ----------------------------------------------------------
    # 检测接口
    # ----------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        执行检测。返回统一的检测结果列表。

        Returns:
            [{"bbox": (x1,y1,x2,y2), "class": str, "confidence": float,
              "source": "yolo"|"motion"|"template", "name": str?}, ...]
        """
        all_detections = []

        t_start = time.perf_counter()

        # YOLO 检测
        if self._rknn:
            dets = self._rknn.detect(frame)
            for d in dets:
                d["source"] = "rknn_yolo"
            all_detections.extend(dets)

        elif self._onnx:
            dets = self._onnx.detect(frame)
            for d in dets:
                d["source"] = "onnx_yolo"
            all_detections.extend(dets)

        elif self._motion:
            dets = self._motion.detect(frame)
            all_detections.extend(dets)

        # 模板匹配 (始终运行，与 YOLO 互补)
        tmpl_results = self._templates.detect(frame)
        for r in tmpl_results:
            all_detections.append({
                "bbox": r["bbox"],
                "class": "ui_element",
                "confidence": r["confidence"],
                "source": "template",
                "name": r["name"],
            })

        self._inference_time = time.perf_counter() - t_start
        self._detect_count += 1

        return all_detections

    def detect_sprites(self, frame: np.ndarray) -> List[dict]:
        """仅返回精灵检测结果 (不含 UI 模板)。"""
        return [d for d in self.detect(frame) if d.get("source") != "template"]

    def detect_ui(self, frame: np.ndarray,
                  names: Optional[List[str]] = None) -> List[dict]:
        """检测 UI 元素。"""
        return self._templates.detect(frame, names=names)

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def backend(self) -> str:
        """当前使用的检测后端。"""
        if self._rknn:
            return "rknn"
        elif self._onnx:
            return "onnx"
        else:
            return "motion"

    @property
    def inference_time(self) -> float:
        return self._inference_time

    @property
    def template_names(self) -> List[str]:
        return self._templates.template_names

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    def release(self):
        if self._rknn:
            self._rknn.release()
        if self._onnx:
            self._onnx.release()

    def __del__(self):
        self.release()


# ============================================================
# 测试
# ============================================================

def test_detector():
    """测试检测器 (使用运动检测模式)。"""
    from utils import setup_logging, load_config
    import time

    setup_logging(level="INFO", log_dir="logs/")

    config = load_config("config.yaml")
    detector = SpriteDetector(config["detector"])

    log.info(f"Using backend: {detector.backend}")
    log.info(f"Loaded templates: {detector.template_names}")

    # 采集测试画面
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera")
        return

    log.info("Press ESC to exit, SPACE to save snapshot")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            detections = detector.detect(frame)
            log.info(f"Found {len(detections)} objects in "
                     f"{detector.inference_time * 1000:.1f}ms")

            # 绘制
            from utils import draw_detections
            vis = draw_detections(frame, detections)

            # 缩小显示
            disp = cv2.resize(vis, (960, 540))
            cv2.imshow("Detector Test", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                cv2.imwrite("detect_snapshot.png", vis)
                log.info("Snapshot saved")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


if __name__ == "__main__":
    test_detector()
