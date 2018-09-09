from itertools import groupby
from pathlib import Path

import cv2


class DataObject:
    def __init__(self, file_id, file_list):
        self.file_id = file_id
        self.parts_files = []
        self.image_file = None
        self.segment_file = None
        self.text_file = None
        for file in file_list:
            if "atr" in str(file.stem):
                self.text_file = file
            elif "seg" in str(file.stem):
                self.segment_file = file
            elif "parts" in str(file.stem):
                self.parts_files.append(file)
            else:
                self.image_file = file

        assert all([val is not None for val in
                    [self.image_file, self.segment_file, self.text_file]])

    def __repr__(self):
        return str(self.file_id)

    @property
    def image(self):
        return self._load_as_numpy(self.image_file)

    @property
    def segment(self):
        return self._load_as_numpy(self.segment_file)

    @property
    def labels(self):
        with open(str(self.text_file), 'r') as file:
            lines = file.readlines()

        labels = []
        for line in lines:
            elements = line.split(" # ")[4:-1]
            labels += elements
        return list(set(labels))

    def _load_as_numpy(self, img_file):
        return cv2.imread(str(img_file))


class Dataset:
    def __init__(self, root_dir):
        dir = Path(root_dir).resolve()
        self.objects = self._get_all_data_objects(dir)

    def __iter__(self):
        for data_object in self.objects:
            yield data_object

    def __len__(self):
        return len(self.objects)

    def _get_all_files(self, path, files):
        """ Get every file in the path recursively """
        for sub_path in path.iterdir():
            if sub_path.is_file():
                files.append(sub_path)
            else:
                self._get_all_files(sub_path, files)

        return files

    def _get_all_data_objects(self, dir):
        """ Return a list of DataObjects """

        all = self._get_all_files(dir, [])

        def get_id(file_name):
            splt = file_name.stem.split("_")
            ints = [x for x in splt if x.isdigit()]
            assert len(ints) >= 1
            return int(ints[0])

        # Group files by the "id" in the filename
        groups = [DataObject(key, list(group))
                  for key, group in
                  groupby(all, key=lambda file_name: get_id(file_name))]
        groups = sorted(groups, key=lambda g: g.file_id)
        return groups


if __name__ == "__main__":
    dataset = Dataset("./images/training/")

    for data in dataset:
        print(data.file_id)
        print(data.image)
        break
