from pathlib import Path
import xml.etree.ElementTree as ET


class Image:
    def __init__(self, img_path, annotation):
        """
        :param id:
        :param annotation:
        :param positive_sets: A list of ("setname", obj_id) tuples
        :param negative_sets: Same as positive_sets, but each obj_id does NOT
                exhibit the behaviour
        """
        self._parse_int = lambda xml_obj, name: \
            int(round(float(xml_obj.find(name).text), 0))

        self.img_path = img_path

        # Parse the annotation to extract objects and image info
        self.label = ET.parse(annotation).getroot()
        self.width = self._parse_int(self.label, "size/width")
        self.height = self._parse_int(self.label, "size/height")
        self.objects = []

        for count, obj in enumerate(self.label.findall("object")):
            self.objects.append(self.extract_object(obj, count))

    def extract_object(self, xml_obj, obj_count):
        name = xml_obj.find("name").text

        rect = [
            self._parse_int(xml_obj, "bndbox/xmin"),
            self._parse_int(xml_obj, "bndbox/ymin"),
            self._parse_int(xml_obj, "bndbox/xmax"),
            self._parse_int(xml_obj, "bndbox/ymax")
        ]

        assert None not in rect

        pos_actions = []
        neg_actions = []
        actions = xml_obj.find("actions")
        if actions is not None:
            # VOC2012 images can sometimes have 'actions' or not have 'actions'
            for action in actions:
                if action.text == "1":
                    pos_actions.append(action.tag)
                elif action.text == "0":
                    neg_actions.append(action.tag)
        elif xml_obj.find("action") is not None:
            # Stanford40 images don't have an "actions" label, but DO have an "action" label
            pos_actions.append(xml_obj.find('action').text)

        return AnnotatedObject(name,
                               obj_count,
                               self.img_path,
                               rect,
                               pos_actions,
                               neg_actions)

    def __repr__(self):
        return self.img_path + " " + self.positive_sets + " " + self.negative_sets

class AnnotatedObject:
    def __init__(self, name, count, img_path, rect, positive_actions, negative_actions):
        """
        :param name: The type of object. person, car, dog
        :param count: The 'id' within the image of the object
        :param img_path: The path to the image that this object belongs in
        :param rect: [x1, y1, x2, y2] of the objects crop
        :param positive_actions: ['running']
        :param negative_actions: ['walking', ..., all other actions]
        """
        self.name = name
        self.count = count
        self.img_path = img_path
        self.rect = rect
        self.positive_actions = positive_actions
        self.negative_actions = negative_actions

    def __repr__(self):
        return "{}(From:{}, Rect:{}, Actions:{})".\
            format(self.name, self.img_path, self.rect, self.positive_actions)

class VOCDataset:
    def __init__(self, img_dir, annotations_dir):
        self.images = self.build_images(img_dir,
                                        annotations_dir)


    def __iter__(self):
        for img in self.images:
            yield img

    def build_images(self, img_dir, annotations_dir):
        """ Return a list of Image objects, fully configured """

        images = []
        for img_path in Path(img_dir).glob("*.jpg"):
            img_id = img_path.stem

            # DEBUG
            # if len(images) >= 100: break

            lbl_path = Path(annotations_dir) / (img_id + ".xml")
            assert lbl_path.exists()

            new_image = Image(str(img_path),
                              lbl_path)
            images.append(new_image)

        return images