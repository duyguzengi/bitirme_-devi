"""
Space ByteTrack Algorithm.
===========================
Adapted for on-orbit space debris monitoring.

Key Mechanism:
Space debris brightness oscillates wildly due to rotational tumbling (light curves)
and varying solar phase angles. Standard trackers discard low-confidence detections,
fragmenting tracklets. ByteTrack uses low-score detections in a second association step
to bridge observation gaps while strictly prohibiting low-score detections from spawning
new false-positive tracklets.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from .kalman_filter import SpaceKalmanFilter


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class SDebrisTrack:
    """Represents a single space debris tracklet with kinematic history."""

    _count = 0

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int = 0):
        SDebrisTrack._count += 1
        self.track_id = SDebrisTrack._count

        self.score = float(score)
        self.class_id = int(class_id)
        self.state = TrackState.New
        self.is_activated = False

        self.kalman_filter = SpaceKalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(tlwh))

        self.tracklet_len = 0
        self.time_since_update = 0

        # Trajectory history: list of [center_x, center_y]
        cx, cy = tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0
        self.history = [(float(cx), float(cy))]
        self.history_len = 30  # Keep last 30 points

    @staticmethod
    def reset_counter():
        SDebrisTrack._count = 0

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3] + 1e-6
        return ret

    def to_tlwh(self) -> np.ndarray:
        if self.mean is None:
            return np.zeros(4, dtype=np.float32)
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    def to_tlbr(self) -> np.ndarray:
        tlwh = self.to_tlwh()
        return np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=np.float32)

    def get_velocity(self) -> Tuple[float, float]:
        """Returns instantaneous velocity (vx, vy) in pixels/sec (assuming 30 fps)."""
        if self.mean is None:
            return 0.0, 0.0
        # Kalman mean state: [x, y, a, h, vx, vy, va, vh]
        vx = float(self.mean[4])
        vy = float(self.mean[5])
        return vx, vy

    def predict(self):
        mean, cov = self.kalman_filter.predict(self.mean, self.covariance)
        self.mean = mean
        self.covariance = cov
        self.time_since_update += 1

    def update(self, tlwh: np.ndarray, score: float):
        self.time_since_update = 0
        self.tracklet_len += 1
        self.score = float(score)
        self.state = TrackState.Tracked
        self.is_activated = True

        measurement = self.tlwh_to_xyah(tlwh)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, measurement)

        cx = float(self.mean[0])
        cy = float(self.mean[1])
        self.history.append((cx, cy))
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


def iou_batch(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two lists of boxes in tlbr format."""
    num_a = len(atlbrs)
    num_b = len(btlbrs)
    if num_a == 0 or num_b == 0:
        return np.zeros((num_a, num_b), dtype=np.float32)

    ious = np.zeros((num_a, num_b), dtype=np.float32)
    for i in range(num_a):
        a = atlbrs[i]
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        for j in range(num_b):
            b = btlbrs[j]
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            xx1 = max(a[0], b[0])
            yy1 = max(a[1], b[1])
            xx2 = min(a[2], b[2])
            yy2 = min(a[3], b[3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            union = area_a + area_b - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious


class SpaceByteTracker:
    """
    ByteTrack implementation specialized for Space Situational Awareness.
    """

    def __init__(
        self,
        track_thresh: float = 0.35,
        low_thresh: float = 0.10,
        match_thresh: float = 0.80,
        frame_rate: int = 30,
        max_time_lost: int = 30,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.max_time_lost = max_time_lost

        self.tracked_tracks: List[SDebrisTrack] = []
        self.lost_tracks: List[SDebrisTrack] = []
        self.removed_tracks: List[SDebrisTrack] = []
        self.frame_id = 0

    def reset(self):
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        SDebrisTrack.reset_counter()

    def update(self, output_results: np.ndarray) -> List[SDebrisTrack]:
        """
        output_results: array of shape (N, 6) where each row is:
            [x1, y1, x2, y2, score, class_id]
        """
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        if len(output_results) > 0:
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]

            # Split into High and Low confidence detections
            remain_inds = scores >= self.track_thresh
            inds_low = (scores < self.track_thresh) & (scores >= self.low_thresh)

            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

            dets_second = bboxes[inds_low]
            scores_second = scores[inds_low]
            classes_second = classes[inds_low]
        else:
            dets = np.empty((0, 4))
            scores_keep = np.empty((0,))
            classes_keep = np.empty((0,))
            dets_second = np.empty((0, 4))
            scores_second = np.empty((0,))
            classes_second = np.empty((0,))

        # ── Step 1: Predict tracks using Kalman filter ───────────────────────
        pool_tracks = self.tracked_tracks + self.lost_tracks
        for t in pool_tracks:
            t.predict()

        # ── Step 2: First association (High Confidence) ──────────────────────
        track_boxes = np.array([t.to_tlbr() for t in pool_tracks]) if pool_tracks else np.empty((0, 4))
        dists = 1.0 - iou_batch(track_boxes, dets)

        matched_a, matched_b = linear_sum_assignment(dists) if dists.size > 0 else ([], [])
        matched_indices = []
        for a, b in zip(matched_a, matched_b):
            if dists[a, b] <= self.match_thresh:
                matched_indices.append((a, b))

        unmatched_tracks_1 = [i for i in range(len(pool_tracks)) if i not in [m[0] for m in matched_indices]]
        unmatched_dets_1 = [i for i in range(len(dets)) if i not in [m[1] for m in matched_indices]]

        for t_idx, d_idx in matched_indices:
            track = pool_tracks[t_idx]
            det = dets[d_idx]
            tlwh = np.array([det[0], det[1], det[2] - det[0], det[3] - det[1]])
            if track.state == TrackState.Tracked:
                track.update(tlwh, scores_keep[d_idx])
                activated_tracks.append(track)
            else:
                track.update(tlwh, scores_keep[d_idx])
                refind_tracks.append(track)

        # ── Step 3: Second association (Low Confidence with Unmatched Tracks) ──
        rem_tracks = [pool_tracks[i] for i in unmatched_tracks_1 if pool_tracks[i].state == TrackState.Tracked]
        if rem_tracks and len(dets_second) > 0:
            rem_boxes = np.array([t.to_tlbr() for t in rem_tracks])
            dists2 = 1.0 - iou_batch(rem_boxes, dets_second)
            matched_a2, matched_b2 = linear_sum_assignment(dists2)
            matched_indices_2 = []
            for a, b in zip(matched_a2, matched_b2):
                if dists2[a, b] <= 0.85:
                    matched_indices_2.append((a, b))

            matched_tracks_2 = set()
            for a, b in matched_indices_2:
                track = rem_tracks[a]
                det = dets_second[b]
                tlwh = np.array([det[0], det[1], det[2] - det[0], det[3] - det[1]])
                track.update(tlwh, scores_second[b])
                activated_tracks.append(track)
                matched_tracks_2.add(a)

            unmatched_tracks_2 = [t for i, t in enumerate(rem_tracks) if i not in matched_tracks_2]
        else:
            unmatched_tracks_2 = rem_tracks

        for track in unmatched_tracks_2:
            track.mark_lost()
            lost_tracks.append(track)

        # ── Step 4: Initialize new tracks from unmatched high-score detections ─
        for d_idx in unmatched_dets_1:
            det = dets[d_idx]
            tlwh = np.array([det[0], det[1], det[2] - det[0], det[3] - det[1]])
            score = scores_keep[d_idx]
            cls_id = classes_keep[d_idx]
            new_track = SDebrisTrack(tlwh, score, cls_id)
            new_track.state = TrackState.Tracked
            new_track.is_activated = True
            activated_tracks.append(new_track)

        # ── Step 5: Update track lists & clean up dead tracks ─────────────────
        self.tracked_tracks = [t for t in activated_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks += refind_tracks

        # Update lost tracks
        for t in lost_tracks:
            if t not in self.lost_tracks:
                self.lost_tracks.append(t)

        # Filter out tracks lost for too long
        cur_lost = []
        for t in self.lost_tracks:
            if t in self.tracked_tracks:
                continue
            if self.frame_id - (t.history_len) > self.max_time_lost or t.time_since_update > self.max_time_lost:
                t.mark_removed()
                removed_tracks.append(t)
            else:
                cur_lost.append(t)
        self.lost_tracks = cur_lost
        self.removed_tracks += removed_tracks

        return [t for t in self.tracked_tracks if t.is_activated]
