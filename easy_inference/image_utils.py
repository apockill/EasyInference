import cv2


def resize_to_fit(img, max_w, max_h, fit_within=True):
    """
    This will resize an image so that the largest image dimension fits within
    the max_w or max_h parameters
    :param img: 3 axis np array
    :param max_w: The desired width, if the image is wider than it is tall
    :param max_h: Desired height, if the image is taller than it is wide
    :param fit_within: If True, the image will fit within max_w, max_h.
           If false, one dimension of image will be larger than max_w, max_h.
    :return:
    """
    cur_h, cur_w, _ = img.shape

    pred_h = (max_w / cur_w) * cur_h

    if fit_within:
        scale_factor = max_h / cur_h if pred_h > max_h else max_w / cur_w
    else:
        scale_factor = max_h / cur_h if pred_h < max_h else max_w / cur_w


    new_w = int(round(cur_w * scale_factor, 0))
    new_h = int(round(cur_h * scale_factor, 0))

    return cv2.resize(img, (new_w, new_h))


def resize_and_pad(img, new_w, new_h, border_color=(0, 0, 0)):
    """
    Resizes an image to the new resolution and keeps the aspect ratio of the
    internal structure of the image, however, it will add a padding of the
    chosen color to make the image fit into the new shape without warping.
    :param img: 3 axis np array
    :param new_w: Desired width
    :param new_h: Desired height
    :param border_color: The color of the borders, if there are any
    :return:
    """
    img = resize_to_fit(img, new_w, new_h)

    cur_h, cur_w, _ = img.shape
    delta_w = new_w - cur_w
    delta_h = new_h - cur_h

    top = delta_h // 2
    left = delta_w // 2
    right, bottom = delta_w - left, delta_h - top

    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT,
                             value=border_color)
    return img


def resize_and_crop(img, new_w, new_h):
    """
    Resize an image and cut off the edges, while minimising information loss.
    No warping.
    :param img: 3 axis np array
    :param new_w: Desired width
    :param new_h: Desired height
    :return:
    """
    img = resize_to_fit(img, new_w, new_h, fit_within=False)

    cur_h, cur_w, _ = img.shape
    delta_w = cur_w - new_w
    delta_h = cur_h - new_h

    x1 = delta_w // 2
    y1 = delta_h // 2

    x2 = cur_w - (delta_w - x1)
    y2 = cur_h - (delta_h - y1)

    cropped = img[y1:y2,
                  x1:x2, :]
    return cropped