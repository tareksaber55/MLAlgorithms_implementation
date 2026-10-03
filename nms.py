import numpy as np

def compute_iou(box, boxes):
    """Compute IoU between a box and a list of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / np.maximum(union_area, 1e-6)


def nms(boxes, scores, iou_thresh = 0.5):
    keep = []
    indices = scores.argsort(descending=True)

    while indices:
        # Take the top-scoring box (the "current" box).
        current = indices[0]
        keep.append(current)
        rest = indices[1:]

        # Filter out any boxes that have IoU ≥ threshold
        # — they overlap too much = likely duplicates.
        ious = compute_iou(boxes[current], boxes[rest])
        indices = rest[ious < iou_thresh]

    return keep



def soft_nms(boxes, scores, iou_thresh=0.5, sigma=0.5, score_thresh=0.001, method='gaussian'):
    """Soft-NMS: boxes (Nx4), scores (N,)"""
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    N = len(scores)
    keep = []

    for i in range(N):
        max_score_idx = i + np.argmax(scores[i:])
        # Swap i-th and max score box
        boxes[[i, max_score_idx]] = boxes[[max_score_idx, i]]
        scores[[i, max_score_idx]] = scores[[max_score_idx, i]]

        box_i = boxes[i]
        for j in range(i+1, N):
            iou = compute_iou(box_i, boxes[j:j+1])[0]

            if method == 'linear':
                if iou > iou_thresh:
                    scores[j] *= (1 - iou)
            elif method == 'gaussian':
                scores[j] *= np.exp(- (iou ** 2) / sigma)
            else:  # hard NMS
                if iou > iou_thresh:
                    scores[j] = 0

    # Filter out low scores
    for i in range(N):
        if scores[i] >= score_thresh:
            keep.append(i)

    return keep, boxes[keep], scores[keep]


if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210],
                      [105, 105, 215, 215],
                      [150, 150, 250, 250]])
    scores = np.array([0.9, 0.8, 0.7])

    keep, final_boxes, final_scores = soft_nms(boxes, scores, method='gaussian')

    print("Final kept indices:", keep)
    print("Final boxes:\n", final_boxes)
    print("Final scores:\n", final_scores)
