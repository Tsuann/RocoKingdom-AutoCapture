"""
目标追踪模块

使用 OpenCV KalmanFilter 对检测到的精灵进行多目标追踪。
- 预测精灵在下一帧的位置
- 平滑运动轨迹，减少抖动
- 处理目标短暂丢失/重新出现
"""

from typing import Dict, List, Optional, Tuple

import threading

import cv2
import numpy as np

from utils import log, bbox_center, distance


class KalmanTracker:
    """
    单个目标的卡尔曼滤波追踪器。

    状态向量: [x, y, vx, vy] (位置 + 速度)
    观测向量: [x, y] (检测到的位置)
    """

    def __init__(self, object_id: int, initial_pos: Tuple[int, int],
                 process_noise: float = 0.03,
                 measurement_noise: float = 0.1):
        self.object_id = object_id
        self.disappeared = 0          # 连续丢失帧数
        self.max_disappeared = 30     # 超时移除

        # 卡尔曼滤波器
        self._kf = cv2.KalmanFilter(4, 2, 0)
        self._init_kalman(initial_pos, process_noise, measurement_noise)

        # 历史轨迹 (用于平滑)
        self.history: List[Tuple[int, int]] = [initial_pos]
        self.max_history = 30

        # 预测位置
        self.predicted: Tuple[int, int] = initial_pos
        self.measured: Tuple[int, int] = initial_pos
        self.smoothed: Tuple[int, int] = initial_pos

    def _init_kalman(self, pos: Tuple[int, int],
                     process_noise: float = 0.03,
                     measurement_noise: float = 0.1):
        """初始化卡尔曼滤波器矩阵。"""
        kf = self._kf

        # 状态转移矩阵 A (匀速模型)
        # x(t+1) = x(t) + vx(t)*dt, vx(t+1) = vx(t)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # 测量矩阵 H (只观测位置)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # 过程噪声协方差 Q
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 测量噪声协方差 R
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # 误差协方差 P (初始不确定性)
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

        # 初始状态
        kf.statePost = np.array([
            [pos[0]], [pos[1]], [0], [0]
        ], dtype=np.float32)

    def predict(self) -> Tuple[int, int]:
        """预测下一帧位置。"""
        pred = self._kf.predict()
        x = int(pred[0, 0])
        y = int(pred[1, 0])
        self.predicted = (x, y)
        return self.predicted

    def update(self, measured_pos: Tuple[int, int]):
        """用新的测量值更新追踪器。"""
        self.disappeared = 0
        self.measured = measured_pos

        measurement = np.array([[measured_pos[0]], [measured_pos[1]]],
                                dtype=np.float32)
        corrected = self._kf.correct(measurement)

        self.smoothed = (int(corrected[0, 0]), int(corrected[1, 0]))
        self.history.append(self.smoothed)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def mark_missed(self):
        """标记本帧未检测到。"""
        self.disappeared += 1
        # 即使没有测量，也用预测值
        self.smoothed = self.predicted

    @property
    def is_lost(self) -> bool:
        """追踪器是否应该被移除。"""
        return self.disappeared > self.max_disappeared

    @property
    def velocity(self) -> Tuple[float, float]:
        """估计速度 (像素/帧)。"""
        state = self._kf.statePost
        return (float(state[2, 0]), float(state[3, 0]))


class MultiTracker:
    """
    多目标追踪管理器。

    用法:
        mt = MultiTracker(config)
        mt.update(detections)
        target = mt.select_target(prefer=["shiny", "corrupted", "normal"])
    """

    def __init__(self,
                 max_disappeared: int = 30,
                 process_noise: float = 0.03,
                 measurement_noise: float = 0.1,
                 iou_match_threshold: float = 0.3):
        self.max_disappeared = max_disappeared
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.iou_match_threshold = iou_match_threshold

        self._trackers: Dict[int, KalmanTracker] = {}
        self._next_id = 0
        self._current_target_id: Optional[int] = None
        self._lock = threading.Lock()

    def update(self, detections: List[dict]) -> List[dict]:
        """线程安全的检测更新。"""
        with self._lock:
            for tracker in self._trackers.values():
                tracker.predict()

            if len(detections) > 0 and len(self._trackers) > 0:
                matches, unmatched_det, unmatched_trk = self._match_iou(detections)
            else:
                matches = []
                unmatched_det = list(range(len(detections)))
                unmatched_trk = list(self._trackers.keys())

            for det_idx, trk_id in matches:
                det = detections[det_idx]
                center = bbox_center(det["bbox"])
                self._trackers[trk_id].update(center)
                det["tracker_id"] = trk_id
                det["predicted_position"] = self._trackers[trk_id].predicted
                det["smoothed_position"] = self._trackers[trk_id].smoothed

            for det_idx in unmatched_det:
                det = detections[det_idx]
                center = bbox_center(det["bbox"])
                trk_id = self._next_id
                self._next_id += 1
                self._trackers[trk_id] = KalmanTracker(
                    trk_id, center, self.process_noise, self.measurement_noise)
                det["tracker_id"] = trk_id
                det["predicted_position"] = center
                det["smoothed_position"] = center

            for trk_id in unmatched_trk:
                self._trackers[trk_id].mark_missed()

            lost_ids = [tid for tid, t in self._trackers.items() if t.is_lost]
            for tid in lost_ids:
                del self._trackers[tid]
                if self._current_target_id == tid:
                    self._current_target_id = None

            return detections

    def get_active_detections(self) -> List[dict]:
        """线程安全地获取活跃追踪器的检测信息（用于显示）。"""
        with self._lock:
            result = []
            for tid, t in self._trackers.items():
                if t.disappeared > 0:
                    continue
                sx, sy = t.smoothed
                result.append({
                    "bbox": (int(sx-30), int(sy-30), int(sx+30), int(sy+30)),
                    "class": "tracked",
                    "confidence": 1.0,
                    "tracker_id": tid,
                })
            return result

    # ----------------------------------------------------------
    # 目标选择
    # ----------------------------------------------------------

    def select_target(self,
                      prefer: Optional[List[str]] = None,
                      detections: Optional[List[dict]] = None) -> Optional[dict]:
        """
        从当前活跃目标中选择最佳目标。

        Args:
            prefer: 类别优先级 (如 ["shiny", "corrupted", "normal"])
            detections: 当前帧检测结果 (含 tracker_id)

        Returns:
            选中的检测结果，或 None
        """
        if not detections:
            detections = []

        # 只考虑有 tracker_id 的检测
        tracked = [d for d in detections if "tracker_id" in d]

        if not tracked:
            return None

        if prefer is None:
            prefer = ["shiny", "corrupted", "normal"]

        # 按优先级排序
        def _priority(det):
            cls = det.get("class", "normal")
            try:
                return prefer.index(cls) if cls in prefer else len(prefer)
            except ValueError:
                return len(prefer)

        tracked.sort(key=lambda d: (_priority(d), -d.get("confidence", 0)))

        best = tracked[0]
        self._current_target_id = best.get("tracker_id")
        return best

    def get_target_position(self) -> Optional[Tuple[int, int]]:
        """获取当前锁定目标的平滑位置。"""
        if self._current_target_id is None:
            return None
        tracker = self._trackers.get(self._current_target_id)
        if tracker is None:
            return None
        return tracker.smoothed

    def get_target_predicted(self) -> Optional[Tuple[int, int]]:
        """获取当前锁定目标的预测位置 (用于提前量瞄准)。"""
        if self._current_target_id is None:
            return None
        tracker = self._trackers.get(self._current_target_id)
        if tracker is None:
            return None
        return tracker.predicted

    # ----------------------------------------------------------
    # 内部匹配
    # ----------------------------------------------------------

    def _match_iou(self, detections: List[dict]) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]
    ]:
        """基于 IoU 的检测-追踪器匹配。"""
        trk_ids = list(self._trackers.keys())

        # 为每个追踪器获取预测 bbox
        trk_bboxes = {}
        for tid in trk_ids:
            tracker = self._trackers[tid]
            px, py = tracker.predicted
            # 用历史中的平均 bbox 大小
            if len(tracker.history) >= 2:
                # 估算运动范围作为 bbox
                pts = np.array(tracker.history[-10:])
                w = np.ptp(pts[:, 0]) + 30
                h = np.ptp(pts[:, 1]) + 30
            else:
                w, h = 60, 60  # 默认大小

            trk_bboxes[tid] = (px - w // 2, py - h // 2, px + w // 2, py + h // 2)

        # 计算 IoU 矩阵
        matches = []
        used_det = set()
        used_trk = set()

        if len(detections) > 0 and len(trk_ids) > 0:
            for d_idx, det in enumerate(detections):
                d_bbox = det["bbox"]

                best_iou = 0.0
                best_tid = None

                for tid in trk_ids:
                    if tid in used_trk:
                        continue
                    t_bbox = trk_bboxes[tid]
                    iou = self._compute_iou(d_bbox, t_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid

                if best_iou >= self.iou_match_threshold and best_tid is not None:
                    matches.append((d_idx, best_tid))
                    used_det.add(d_idx)
                    used_trk.add(best_tid)

        unmatched_det = [i for i in range(len(detections)) if i not in used_det]
        unmatched_trk = [tid for tid in trk_ids if tid not in used_trk]

        return matches, unmatched_det, unmatched_trk

    def _compute_iou(self, bbox_a: Tuple[int, int, int, int],
                     bbox_b: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的 IoU。"""
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
        area_b = max(0, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._trackers)

    @property
    def current_target_id(self) -> Optional[int]:
        return self._current_target_id

    def reset(self):
        """清除所有追踪器。"""
        self._trackers.clear()
        self._current_target_id = None

    def get_info(self) -> dict:
        """获取当前追踪状态。"""
        target_tracker = None
        if self._current_target_id is not None:
            target_tracker = self._trackers.get(self._current_target_id)

        return {
            "active_trackers": self.active_count,
            "current_target": self._current_target_id,
            "target_position": target_tracker.smoothed if target_tracker else None,
            "target_velocity": target_tracker.velocity if target_tracker else None,
        }
