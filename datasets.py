import torch


class ScansDataset(torch.utils.data.Dataset):
    """CM scans dataset with possibility to (linearly) stain."""

    def __init__(self, root_dir, stain=True, transform_stained=None, transform_F=None, transform_R=None):
        """
        Args:
            root_dir: str. Directory with "mosaics"
            stain: bool. Stain CM image using VirtualStainer
            scale_by: int. Divide pixel values prior to staining.
            transform_stained: callable object. Apply transform to stained image.
            transform_F: callable object. Apply transform to F-mode image.
            transform_R: callable object. Apply transform to R-mode image.
        """
        self.root_dir = root_dir
        self.transform_stained = transform_stained
        self.transform_F = transform_F
        self.transform_R = transform_R
        self.scans = self._list_scans()
        if stain:
            self.stainer = VirtualStainer()

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, item):
        """Get CM image.
        If stain, return stained image. Return both modes otherwise.
        """
        f_file = self.scans[item] + '/DET#2/highres_raw.tif'
        r_file = self.scans[item] + '/DET#1/highres_raw.tif'
        f_img = pyvips.Image.new_from_file(f_file, access='sequential')
        r_img = pyvips.Image.new_from_file(r_file, access='sequential')

        if self.transform_F:
            f_img = self.transform_F(f_img)
        if self.transform_R:
            r_img = self.transform_R(r_img)
        if self.stainer:
            img = self.stainer(f_img, r_img)
            if self.transform_stained:
                return self.transform_stained(img)
            return img
        return {'F': f_img, 'R': r_img}

    def _list_scans(self):
        scans = []
        for root, dirs, files in os.walk(self.root_dir):
            if 'mosaic' in root.split('/')[-1]:
                scans.append(root)

        scans = sorted(list(set(scans)))
        return scans

