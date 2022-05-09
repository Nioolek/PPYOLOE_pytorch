import uuid
import numpy as np


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    # def apply(self, sample, targets, context=None):
    #     """ Process a sample.
    #     Args:
    #         sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
    #         context (dict): info about this sample processing
    #     Returns:
    #         result (dict): a processed sample
    #     """
    #     return sample, targets
    #
    # def __call__(self, sample, targets, context=None):
    #     """ Process a sample.
    #     Args:
    #         sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
    #         context (dict): info about this sample processing
    #     Returns:
    #         result (dict): a processed sample
    #     """
    #     if isinstance(sample, list):
    #         for i in range(len(sample)):
    #             sample[i] = self.apply(sample[i],targets, context)
    #     else:
    #         sample = self.apply(sample, targets, context)
    #     return sample, targets
    #
    # def __str__(self):
    #     return str(self._id)


class RandomDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        return img

    def apply(self, sample, targets, context=None):
        img = sample.copy()
        # img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            # sample['image'] = img
            return img, targets

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        img = self.apply_saturation(img)
        img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        # sample['image'] = img
        return img, targets


class Pad(BaseOperator):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, Sequence): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """
        super(Pad, self).__init__()

        if not isinstance(size, (int, list)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    # def apply_segm(self, segms, offsets, im_size, size):
    #     def _expand_poly(poly, x, y):
    #         expanded_poly = np.array(poly)
    #         expanded_poly[0::2] += x
    #         expanded_poly[1::2] += y
    #         return expanded_poly.tolist()
    #
    #     def _expand_rle(rle, x, y, height, width, h, w):
    #         if 'counts' in rle and type(rle['counts']) == list:
    #             rle = mask_util.frPyObjects(rle, height, width)
    #         mask = mask_util.decode(rle)
    #         expanded_mask = np.full((h, w), 0).astype(mask.dtype)
    #         expanded_mask[y:y + height, x:x + width] = mask
    #         rle = mask_util.encode(
    #             np.array(
    #                 expanded_mask, order='F', dtype=np.uint8))
    #         return rle
    #
    #     x, y = offsets
    #     height, width = im_size
    #     h, w = size
    #     expanded_segms = []
    #     for segm in segms:
    #         if is_poly(segm):
    #             # Polygon format
    #             expanded_segms.append(
    #                 [_expand_poly(poly, x, y) for poly in segm])
    #         else:
    #             # RLE format
    #             import pycocotools.mask as mask_util
    #             expanded_segms.append(
    #                 _expand_rle(segm, x, y, height, width, h, w))
    #     return expanded_segms

    def apply_bbox(self, bbox, offsets):
        # 这里的*2是把列表长度*2
        bbox[:, :4] += np.array(offsets * 2, dtype=np.float32)
        return bbox

    # def apply_keypoint(self, keypoints, offsets):
    #     n = len(keypoints[0]) // 2
    #     return keypoints + np.array(offsets * n, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def apply(self, sample, targets, context=None):
        # print('444', type(sample), type(targets))
        im = sample.copy()
        targets = targets.copy()
        # im = sample['image']
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert (
                im_h < h and im_w < w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            # print('555', type(sample), type(targets))
            return im, targets

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        im = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            # print('666', type(sample), type(targets))
            return im, targets
        targets = self.apply_bbox(targets, offsets)
        # if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
        #     sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)
        #
        # if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
        #     sample['gt_poly'] = self.apply_segm(sample['gt_poly'], offsets,
        #                                         im_size, size)
        #
        # if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
        #     sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'],
        #                                                 offsets)
        # print('777', type(sample), type(targets))
        return im, targets


class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (tuple): color value used to fill the canvas. in RGB order.
    """

    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5, 127.5, 127.5)):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, tuple)
        # assert isinstance(fill_value, (Number, Sequence)), \
        #     "fill value must be either float or sequence"
        # if isinstance(fill_value, Number):
        #     fill_value = (fill_value, ) * 3
        # if not isinstance(fill_value, tuple):
        #     fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def apply(self, sample, targets, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            # print('111', type(sample), type(targets))
            return sample, targets

        img = sample.copy()
        height, width = img.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            # print('222', type(sample), type(targets))
            return img, targets
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        pad = Pad(size,
                  pad_mode=-1,
                  offsets=offsets,
                  fill_value=self.fill_value)
        # print('333', type(sample), type(targets))
        return pad.apply(sample, targets, context=context)


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    # def crop_segms(self, segms, valid_ids, crop, height, width):
    #     def _crop_poly(segm, crop):
    #         xmin, ymin, xmax, ymax = crop
    #         crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
    #         crop_p = np.array(crop_coord).reshape(4, 2)
    #         crop_p = Polygon(crop_p)
    #
    #         crop_segm = list()
    #         for poly in segm:
    #             poly = np.array(poly).reshape(len(poly) // 2, 2)
    #             polygon = Polygon(poly)
    #             if not polygon.is_valid:
    #                 exterior = polygon.exterior
    #                 multi_lines = exterior.intersection(exterior)
    #                 polygons = shapely.ops.polygonize(multi_lines)
    #                 polygon = MultiPolygon(polygons)
    #             multi_polygon = list()
    #             if isinstance(polygon, MultiPolygon):
    #                 multi_polygon = copy.deepcopy(polygon)
    #             else:
    #                 multi_polygon.append(copy.deepcopy(polygon))
    #             for per_polygon in multi_polygon:
    #                 inter = per_polygon.intersection(crop_p)
    #                 if not inter:
    #                     continue
    #                 if isinstance(inter, (MultiPolygon, GeometryCollection)):
    #                     for part in inter:
    #                         if not isinstance(part, Polygon):
    #                             continue
    #                         part = np.squeeze(
    #                             np.array(part.exterior.coords[:-1]).reshape(1,
    #                                                                         -1))
    #                         part[0::2] -= xmin
    #                         part[1::2] -= ymin
    #                         crop_segm.append(part.tolist())
    #                 elif isinstance(inter, Polygon):
    #                     crop_poly = np.squeeze(
    #                         np.array(inter.exterior.coords[:-1]).reshape(1, -1))
    #                     crop_poly[0::2] -= xmin
    #                     crop_poly[1::2] -= ymin
    #                     crop_segm.append(crop_poly.tolist())
    #                 else:
    #                     continue
    #         return crop_segm
    #
    #     def _crop_rle(rle, crop, height, width):
    #         if 'counts' in rle and type(rle['counts']) == list:
    #             rle = mask_util.frPyObjects(rle, height, width)
    #         mask = mask_util.decode(rle)
    #         mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
    #         rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    #         return rle
    #
    #     crop_segms = []
    #     for id in valid_ids:
    #         segm = segms[id]
    #         if is_poly(segm):
    #             import copy
    #             import shapely.ops
    #             from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    #             logging.getLogger("shapely").setLevel(logging.WARNING)
    #             # Polygon format
    #             crop_segms.append(_crop_poly(segm, crop))
    #         else:
    #             # RLE format
    #             import pycocotools.mask as mask_util
    #             crop_segms.append(_crop_rle(segm, crop, height, width))
    #     return crop_segms

    def apply(self, sample, targets, context=None):
        if len(targets) == 0:
            return sample, targets
        # print('type1', type(sample), type(targets))
        img, targets = sample.copy(), targets.copy()

        h, w = img.shape[:2]
        gt_bbox = targets[:, :4]

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return img, targets

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(
                        max(min_ar, scale**2), min(max_ar, scale**-2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                # if self.is_mask_crop and 'gt_poly' in sample and len(sample[
                #         'gt_poly']) > 0:
                #     crop_polys = self.crop_segms(
                #         sample['gt_poly'],
                #         valid_ids,
                #         np.array(
                #             crop_box, dtype=np.int64),
                #         h,
                #         w)
                #     if [] in crop_polys:
                #         delete_id = list()
                #         valid_polys = list()
                #         for id, crop_poly in enumerate(crop_polys):
                #             if crop_poly == []:
                #                 delete_id.append(id)
                #             else:
                #                 valid_polys.append(crop_poly)
                #         valid_ids = np.delete(valid_ids, delete_id)
                #         if len(valid_polys) == 0:
                #             return sample
                #         sample['gt_poly'] = valid_polys
                #     else:
                #         sample['gt_poly'] = crop_polys

                # if 'gt_segm' in sample:
                #     sample['gt_segm'] = self._crop_segm(sample['gt_segm'],
                #                                         crop_box)
                #     sample['gt_segm'] = np.take(
                #         sample['gt_segm'], valid_ids, axis=0)

                img = self._crop_image(img, crop_box)
                gt_bbox = np.take(cropped_box, valid_ids, axis=0)
                gt_class = np.take(targets[:, 4:], valid_ids, axis=0)
                targets = np.concatenate((gt_bbox, gt_class), axis=-1)
                # if 'gt_score' in sample:
                #     sample['gt_score'] = np.take(
                #         sample['gt_score'], valid_ids, axis=0)

                # if 'is_crowd' in sample:
                #     sample['is_crowd'] = np.take(
                #         sample['is_crowd'], valid_ids, axis=0)

                # if 'difficult' in sample:
                #     sample['difficult'] = np.take(
                #         sample['difficult'], valid_ids, axis=0)

                return img, targets

        return img, targets

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

    def _crop_segm(self, segm, crop):
        x1, y1, x2, y2 = crop
        return segm[:, y1:y2, x1:x2]


class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply(self, sample, targets, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        im = sample.copy()
        targets = targets.copy()
        if np.random.uniform(0, 1) < self.prob:
            # im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            targets[:, :4] = self.apply_bbox(targets[:, :4], width)

        return im, targets

