import cv2


def resize_and_crop(img, output_side_length, interpolation=cv2.INTER_AREA):
    """ Takes an image name, resize it and crop the center square """
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * (height / width)
    else:
        new_width = output_side_length * (width / height)

    resized_img = cv2.resize(img, (int(new_width), int(new_height)), interpolation=interpolation)
    height_offset = int((new_height - output_side_length) / 2)
    width_offset = int((new_width - output_side_length) / 2)

    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]

    return cropped_img