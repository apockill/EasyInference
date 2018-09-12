from argparse import ArgumentParser

import ujson

from dataset_utils.coco.dataset import COCODataset


class AttributesDataset(COCODataset):
    """For more info:
    https://github.com/genp/cocottributes/blob/master/DATA.md"""

    def __init__(self, coco_attributes_file, coco_dir, train=True):
        super().__init__(coco_dir, train=train)

        def intify_keys(dict_type):
            return dict(zip(map(int, dict_type.keys()),
                            dict_type.values()))

        self.attr_data = ujson.load(open(coco_attributes_file, 'r'))
        self.attr_vecs = intify_keys(self.attr_data["ann_vecs"])
        self.attr_details = self.attr_data["attributes"]

        self.attr_names = [item['name'] for item in self.attr_details]
        self.attr_id_to_ann_id = intify_keys(
            self.attr_data["patch_id_to_ann_id"])

        # self.ann_id_to_attr_id = dict(zip(self.attr_id_to_ann_id.values(),
        #                                   self.attr_id_to_ann_id.keys()))
        # self.ann_id_to_attr_id = intify_keys(self.ann_id_to_attr_id)
        # #
        # # print("ayy", self.attr_data.keys(), self.attr_data["split"])

    def __iter__(self):
        print("ayy", len(self.attr_vecs),
              len(self.attr_data["patch_id_to_ann_id"].keys()),
              len(self.coco_annotations))

        # images = list(super().__iter__())
        # id_img_map = {img.id:img for img in images}
        ann_id_to_an = {ann["id"]: ann for ann in self.coco_annotations}

        for key in list(self.attr_vecs.keys()):
            attr_vec = self.attr_vecs[key]
            ann_id = self.attr_id_to_ann_id[key]
            if ann_id not in ann_id_to_an:
                continue
            ann = ann_id_to_an[ann_id]
            print(attr_vec[0], ann_id, ann["id"], ann["image_id"])

            # ann_id = self.ann_id_to_attr_id[idx]
            # anns = [a for a in annotations['annotations'] if a['id'] == ann_id]
            # # anns should have only one entry since id is unique
            # ann = anns[0]
            # # Get the image name
            # img_id = ann['image_id']
            #
            # for index, annotation in enumerate(self.coco_annotations):
            #     ann_id = self.coco_annotations[index]["id"]
            #
            #     attr = self.attr_details['ann_attrs'][int(ann_id)]
            #     attr_vector = attr["attrs_vector"]
            #     # anns = [a for a in self.coco_annotations if a['id'] == ann_id]
            #     print("Got thing")
            #
            #
            # for image_info in super().__iter__():
            #     for annotation in image_info.annotations:
            #         ann_id = self.attr_id_to_ann_id[annotation.id]
            #         anns = [a for a in self.coco_annotations if a['id'] == ann_id]
            #     #     if image_info.id not in self.ann_id_to_attr_id.keys():
            #     #         print("Skipped")
            #     #         continue
            #     #     coco_attr_id = self.ann_id_to_attr_id[image_info.id]
            #     #
            #     #     if coco_attr_id not in self.attr_vecs:
            #     #         print("Skipped")
            #     #         continue
            #     #     # print("Ayylmao")
            #     #     instance_attrs = self.attr_vecs[coco_attr_id]
            #     # # print("attr", attributes)
            #     yield image_info


def main(args):
    print("Loading dataset...")
    attr_dataset = AttributesDataset(args.attributes_file, args.coco_dir,
                                     train=True)

    print("Iterating dataset...")
    for annotation in attr_dataset:
        pass


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Helps you navigate the COCO attributes dataset")
    parser.add_argument("-c", "--coco_dir", type=str, required=True,
                        help="The coco dataset directory")
    parser.add_argument("-a", "--attributes_file", type=str, default=None,
                        help="The coco attributes file, " \
                             "if using COCO attributes dataset")
    args = parser.parse_args()

    main(args)
