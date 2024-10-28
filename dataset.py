import torch
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from option import get_option
import json
import os.path as osp

opt = get_option()

label_mapping = {"line": 0}


class Identity(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Identity, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return img


train_transform = A.Compose(
    [
        A.Resize(opt.image_size, opt.image_size),
        # A.RandomResizedCrop(
        #     opt.image_size,
        #     opt.image_size,
        #     scale=(0.64, 1.0),
        # ),
        # A.D4(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
        # A.SomeOf(
        #     [
        #         ## Color
        #         A.SomeOf(
        #             [
        #                 A.Sharpen(),
        #                 A.Posterize(),
        #                 A.RandomBrightnessContrast(),
        #                 A.RandomGamma(),
        #                 A.ColorJitter(),
        #             ],
        #             n=2,
        #         ),
        #         ## CLAHE
        #         A.CLAHE(),
        #         ## Noise
        #         A.GaussNoise(),
        #         ## Blur
        #         A.AdvancedBlur(),
        #         ## Others
        #         A.ToGray(),
        #         Identity(),
        #     ],
        #     n=opt.aug_m,
        # ),
        A.Normalize(),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", min_visibility=0.1, label_fields=["class_labels"]
    ),
)

valid_transform = A.Compose(
    [
        A.Resize(opt.image_size, opt.image_size),
        A.Normalize(),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", min_visibility=0.1, label_fields=["class_labels"]
    ),
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.data_path = opt.data_path
        self.image_path = opt.image_path
        self.transform = train_transform if phase == "train" else valid_transform
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        self.image_list = list(self.data)

    def __getitem__(self, index):
        image_key = self.image_list[index]
        image_path = osp.join(self.image_path, self.data[image_key]["image_name"])
        bbox = self.data[image_key]["bbox"]
        label = self.data[image_key]["label"]
        image = np.array(Image.open(image_path))
        transformed = self.transform(image=image, bboxes=bbox, class_labels=label)
        image, bbox, label = (
            transformed["image"],
            transformed["bboxes"],
            transformed["class_labels"],
        )
        return {"image": image, "bbox": bbox, "label": label}

    def __len__(self):
        return len(self.data)

    def load_image(self, path):
        return None

    def load_images_in_parallel(self):
        with ThreadPoolExecutor(max_workers=24) as executor:
            pass


def dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for data in batch:
        images.append(data["image"])
        bboxes.append(torch.tensor(data["bbox"]))
        labels.append(
            torch.tensor([label_mapping.get(label) for label in data["label"]])
        )
    images = torch.stack(images)
    return images, bboxes, labels


def get_dataloader(opt):
    train_dataset = Dataset(
        phase="train",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    valid_dataset = Dataset(
        phase="valid",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=dataset_collate,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=dataset_collate,
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    for i, batch in enumerate(train_dataloader):
        print(batch[0].shape)
        break
