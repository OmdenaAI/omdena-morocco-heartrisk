from pathlib import Path

from google.cloud import storage


class GCloudStorage(object):
    def __init__(self, bucket_name, downloads_directory="data"):
        self.bucket_name = bucket_name
        self.downloads_directory = downloads_directory
        self.bucket = storage.Client().get_bucket(self.bucket_name)

    def download_file(self, filename):
        local_path = self.bucket_path_to_local_path(filename)
        self._create_parent_directory_if_missing(local_path)

        blob = self.bucket.get_blob(filename)
        blob.download_to_filename(local_path)

    def bucket_path_to_local_path(self, bucket_filename) -> Path:
        return Path(self.downloads_directory).joinpath(bucket_filename)

    @staticmethod
    def _create_parent_directory_if_missing(file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
