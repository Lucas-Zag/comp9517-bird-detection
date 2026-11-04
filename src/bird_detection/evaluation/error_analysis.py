from .iou import calculate_iou


def evaluate_image(result, label_path, iou_threshold=0.5):
    """Match predicted boxes to ground truth for one image."""

    H, W = result.orig_shape

    # Load YOLO ground-truth annotations
    gt_boxes = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        cls, xc, yc, w, h = map(float, line.split())

        gt_boxes.append([
            (xc - w / 2) * W,
            (yc - h / 2) * H,
            (xc + w / 2) * W,
            (yc + h / 2) * H,
        ])

    # Predicted boxes
    pred_boxes = result.boxes.xyxy.cpu().tolist()

    # Candidate matches above IoU threshold
    candidates = []

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou = calculate_iou(gt, pred)

            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    # Greedy one-to-one matching
    matched_gt = set()
    matched_pred = set()

    for iou, i, j in sorted(candidates, reverse=True):
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 1.0

    return {
        "image": result.path,
        "gt_boxes": gt_boxes,
        "pred_boxes": pred_boxes,
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "gt_count": len(gt_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1,
    }
