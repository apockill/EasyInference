from argparse import ArgumentParser
from pathlib import Path
from threading import Thread
from queue import Queue

import cv2

from easy_inference.image_utils import resize_and_crop

from celeba import CelebA, Image


def _worker(min_size, write_to, work_queue: Queue):
    while not work_queue.empty():
        img: Image = work_queue.get()
        if min_size is not None:
            h = img.rect[3] - img.rect[1]
            w = img.rect[2] - img.rect[0]

            if h < min_size and w < min_size:
                continue
        frame = img.frame

        face = frame[img.rect[1]:img.rect[3],
               img.rect[0]:img.rect[2]]

        if min_size is not None:
            h, w, _ = face.shape
            if h < min_size and w < min_size:
                raise ValueError(h, w, "WHOA?")

            # Resize and crop the image to a square
            face = resize_and_crop(face, min_size, min_size)

        # Verify the shape of the image is correct
        assert all(s == min_size for s in face.shape[:2])

        cv2.imwrite(str(Path(write_to) / img.image_path.name), face)


def crop_images(dataset, write_to, min_size=None, num_workers=32):
    Path(write_to).mkdir(parents=True, exist_ok=True)

    # Create work to be done
    work_queue = Queue()
    for img in dataset:
        work_queue.put(img)

    # Create workers and start their threads
    workers = [Thread(target=_worker,
                      args=(min_size, write_to, work_queue))
               for _ in range(num_workers)]
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    print("Finished processing!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--unaligned_imgs_dir", required=True,
                        help="Path to the images directory")
    parser.add_argument("--attr_dir", required=True,
                        help="Path to the directory with the various label"
                             " files")
    parser.add_argument("--copy_to", required=True,
                        help="Path to copy new (cropped) images")
    parser.add_argument("--resize_to", type=int, default=None,
                        help="If this argument is set, after cropping images"
                             " will be resized to a standard size")
    args = parser.parse_args()

    dataset = CelebA(images_dir=args.unaligned_imgs_dir,
                     attr_dir=args.attr_dir)
    crop_images(dataset=dataset,
                min_size=args.resize_to,
                write_to=args.copy_to)
