def bb_intersection_over_union(box_a, box_b):
    """Thank you pyimagesearch!
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    :param box_a: Rect
    :param box_b: Rect
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    bbox_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    bbox_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(bbox_a_area + bbox_b_area - inter_area)

    # return the intersection over union value
    return iou
