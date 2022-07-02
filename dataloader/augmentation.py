import numpy as np
import cv2


class BaseTransform(object):
    def __init__(self, size=300):
        self.size = size
        self.mean = np.array([123, 117, 104], dtype=np.float32)

    def __call__(self, img, boxes, labels):
        img = cv2.resize(img, dsize=(self.size, self.size))
        img -= self.mean
        return img, boxes, labels


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes, labels):
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels


class ConvertIntToFloat(object):
    def __call__(self, img, boxes, labels):
        return img.astype(np.float32), boxes, labels


class SubtractMean(object):
    def __init__(self):
        self.mean = np.array([123, 117, 104], dtype=np.float32)

    def __call__(self, img, boxes, labels):
        img -= self.mean
        return img, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, img, boxes, labels):
        h, w, _ = img.shape
        boxes[:, (0, 2)] *= w
        boxes[:, (1, 3)] *= h
        return img, boxes, labels


class ToRelativeCoords(object):
    def __call__(self, img, boxes, labels):
        h, w, _ = img.shape
        boxes[:, (0, 2)] /= w
        boxes[:, (1, 3)] /= h
        return img, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, img, boxes, labels):
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(self.size, self.size))
        return img, boxes, labels


class SwapChannels(object):
    def __init__(self):
        self.swaps = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 1, 0),
            (2, 0, 1),
        ]

    def __call__(self, img, boxes, labels):
        swap = self.swaps[np.random.randint(0, len(self.swaps))]
        img = img[:, :, swap]
        return img, boxes, labels


class ConvertColorSpace(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __call__(self, img, boxes, labels):
        if self.start == "RGB" and self.end == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.start == "HSV" and self.end == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            raise ValueError("Not Support.")
        return img, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, boxes=None, labels=None):
        if np.random.randint(2):
            img[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return img, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img, boxes=None, labels=None):
        if np.random.randint(2):
            img[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            img *= alpha
        return img, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            img += delta
        return img, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColorSpace(start="RGB", end="HSV"),
            RandomSaturation(),
            RandomHue(),
            ConvertColorSpace(start="HSV", end="RGB"),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = SwapChannels()

    def __call__(self, img, boxes, labels):
        img, boxes, labels = self.rand_brightness(img, boxes, labels)
        distort = (
            Compose(self.pd[:-1]) if np.random.randint(2) else Compose(self.pd[1:])
        )
        img, boxes, labels = distort(img, boxes, labels)
        return self.rand_light_noise(img, boxes, labels)


class RandomExpand(object):
    def __call__(self, img, boxes, labels):
        if np.random.randint(2):
            return img, boxes, labels
        h, w, _ = img.shape
        scale_ratio = np.random.uniform(1, 4)
        img_left = int(np.random.uniform(0, w * scale_ratio - w))
        img_top = int(np.random.uniform(0, h * scale_ratio - h))

        expand_img = np.zeros(
            (int(h * scale_ratio), int(w * scale_ratio), 3),
            dtype=np.float32,
        )
        expand_img[:, :] = np.array([123, 117, 104], dtype=np.float32)
        expand_img[img_top : img_top + h, img_left : img_left + w, :] = img
        img = expand_img

        boxes[:, (0, 2)] += img_left
        boxes[:, (1, 3)] += img_top
        return img, boxes, labels


class RandomFlip(object):
    def __call__(self, img, boxes, labels):
        if np.random.randint(2):
            _, w, _ = img.shape
            img = img[:, ::-1, :]
            boxes[:, 0::2] = w - boxes[:, 2::-2]
        return img, boxes, labels


class MultipleTransform(object):
    def __init__(self, size):
        self.size = size
        self.augment = Compose(
            [
                ConvertIntToFloat(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                RandomExpand(),
                RandomFlip(),
                ToRelativeCoords(),
                Resize(self.size),
                SubtractMean(),
            ]
        )

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
