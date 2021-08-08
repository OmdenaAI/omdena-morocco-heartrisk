import ast
from pathlib import Path

import imageio
import wfdb

import numpy as np
import pandas as pd

from ecg_image_generator import ECGImageGenerator

from gcloud_storage import GCloudStorage
from skimage.filters import threshold_otsu


class PtbXlEcgManager(object):
    BUCKET_NAME = "ptb-xl-1.0.1.physionet.org"
    DOWNLOADS_DIRECTORY = "ptb_xl_data"

    def __init__(self, downloads_directory=None, download_if_missing=True, generate_images_if_missing=True):
        self.downloads_directory = downloads_directory or self.DOWNLOADS_DIRECTORY

        self.gcloud_storage = GCloudStorage(
            self.BUCKET_NAME, self.downloads_directory
        )
        self._load_ecg_labels()

        self.download_if_missing = download_if_missing
        self.generate_images_if_missing = generate_images_if_missing

    def load_ecg(self, ecg_id):
        ecg_path = self.ecg_labels.loc[ecg_id, "filename_lr"]

        if not self._is_ecg_downloaded(ecg_id):
            if not self.download_if_missing:
                raise RuntimeError(f"The ecg with id {ecg_id} is not downloaded. Download it first or set"
                                   f"the 'download_if_missing' flag to True.")

            self.download_ecg(ecg_id)

        ecg, ecg_metadata = wfdb.rdsamp(
            str(self.gcloud_storage.bucket_path_to_local_path(ecg_path))
        )

        return ecg, ecg_metadata

    def get_ecg_image(self, ecg_id):
        self._generate_image_and_mask_if_missing(ecg_id)

        image_path = self._ecg_image_path(ecg_id)
        return np.array(imageio.imread(image_path, pilmode="RGB", as_gray=True)) / 255

    def get_ecg_mask(self, ecg_id):
        self._generate_image_and_mask_if_missing(ecg_id)

        return np.load(self._ecg_mask_path(ecg_id))

    def plot_ecg(self, ecg_id, save=False, save_mask=False, **kwargs):
        ecg, ecg_metadata = self.load_ecg(ecg_id=ecg_id)

        image_generator = ECGImageGenerator(**kwargs)

        if not save:
            return image_generator.plot_ecg(ecg, ecg_metadata)

        fig = image_generator.plot_ecg(ecg, ecg_metadata, output=self._ecg_image_path(ecg_id))

        if not save_mask:
            return fig

        figure_clean = image_generator.plot_ecg(ecg, ecg_metadata, clean_generation=True)

        img_bytes = figure_clean.to_image(format="png")
        image = imageio.imread(img_bytes, pilmode="RGB", as_gray=True)
        threshold = threshold_otsu(image)
        binary = image < threshold

        np.save(self._ecg_mask_path(ecg_id), binary)

        return fig

    def _load_ecg_labels(self):
        files = ["ptbxl_database.csv", "scp_statements.csv"]
        for file in files:
            if not self.gcloud_storage.bucket_path_to_local_path(file).is_file():
                self.gcloud_storage.download_file(file)

        ecg_labels = pd.read_csv(f"{self.downloads_directory}/ptbxl_database.csv", index_col='ecg_id')
        ecg_labels.recording_date = pd.to_datetime(ecg_labels.recording_date)
        ecg_labels.scp_codes = ecg_labels.scp_codes.apply(lambda x: ast.literal_eval(x))

        self.ecg_labels = ecg_labels

    def _is_ecg_downloaded(self, ecg_id):
        ecg_local_path = self._ecg_local_path_no_suffix(ecg_id)
        return ecg_local_path.with_suffix(".hea").is_file() and ecg_local_path.with_suffix(".dat").is_file()

    def _is_ecg_image_and_mask_generated(self, ecg_id):
        image_path = self._ecg_image_path(ecg_id)
        mask_path = self._ecg_mask_path(ecg_id)
        return image_path.is_file() and mask_path.is_file()

    def download_ecg(self, ecg_id):
        ecg_path = self._ecg_path_no_suffix(ecg_id)
        self.gcloud_storage.download_file(f"{ecg_path}.hea")
        self.gcloud_storage.download_file(f"{ecg_path}.dat")

    def _generate_image_and_mask_if_missing(self, ecg_id):
        if not self._is_ecg_image_and_mask_generated(ecg_id):
            if not self.generate_images_if_missing:
                raise RuntimeError("Corresponding images have not been generated")

            if not self._is_ecg_downloaded(ecg_id):
                if not self.download_if_missing:
                    raise RuntimeError(f"The ecg with id {ecg_id} is not downloaded. Download it first or set"
                                   f"the 'download_if_missing' flag to True.")

                self.download_ecg(ecg_id)

            self.plot_ecg(ecg_id, save=True, save_mask=True);

    def _ecg_path_no_suffix(self, ecg_id):
        return self.ecg_labels.loc[ecg_id, "filename_lr"]

    def _ecg_local_path_no_suffix(self, ecg_id):
        return Path(self.downloads_directory).joinpath(self._ecg_path_no_suffix(ecg_id))

    def _ecg_image_path(self, ecg_id):
        return self._ecg_local_path_no_suffix(ecg_id).with_suffix(".png")

    def _ecg_mask_path(self, ecg_id):
        ecg_path = self._ecg_local_path_no_suffix(ecg_id)
        return ecg_path.parent.joinpath(f"{ecg_path.stem}_mask").with_suffix(".npy")
