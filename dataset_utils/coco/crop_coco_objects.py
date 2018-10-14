from argparse import ArgumentParser
from pathlib import Path
from threading import Thread
from queue import Queue

import cv2

from dataset_utils.coco.dataset import COCODataset


def worker_thread(work_queue: Queue, output_dir):
    while True:
        annotation, image = work_queue.get()
        area = int(annotation.width * annotation.height)
        if area < args.min_area:
            continue

        # Check if this image has already been cropped- and skip if it haspp
        save_to = Path(args.output_dir) / annotation.cat_name
        save_to.mkdir(parents=True, exist_ok=True)
        save_to = save_to / \
                  (str(image.id) + "_" + str(annotation.id) + ".jpg")
        if save_to.is_file():
            continue

        frame = image.load_frame()
        crop = annotation.crop(frame)

        cv2.imwrite(str(save_to), crop)
        work_queue.task_done()


def main(dataset, categories, num_workers, output_dir):
    # Start worker threads
    work_queue = Queue()
    for _ in range(num_workers):
        w = Thread(target=worker_thread,
                   args=(work_queue, output_dir),
                   daemon=True)
        w.start()


    # Create work
    dataset = COCODataset(coco_dir=dataset)
    for image in dataset:
        annotations = image.filter_annotations(category_names=categories)

        if len(annotations) == 0:
            continue

        for annotation in annotations:
            work_queue.put((annotation, image))

    # Wait for work to finish
    work_queue.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to the coco dataset")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to save cropped images to")
    parser.add_argument("--categories", "-c", nargs="+", required=True,
                        help="The categories to crop")
    parser.add_argument("--min_area", "-a", type=int, required=True,
                        help="The minimum pixel area for this to be included")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="The number of worker threads for this task")
    args = parser.parse_args()
    main(dataset=args.dataset,
         categories=args.categories,
         num_workers=args.num_workers,
         output_dir=args.output_dir)
