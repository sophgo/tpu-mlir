import zipfile
import numpy as np
from numpy.lib import format


class IncNpzFile:

    def __init__(self, file: str):
        """
        :param file: the ``npz`` file to write
        :param mode: must be one of {'x', 'w', 'a'}. See
               https://docs.python.org/3/library/zipfile.html for detail
        """
        self.fn = file
        self.zip = zipfile.ZipFile(file, mode="a", compression=zipfile.ZIP_DEFLATED)
        self.keys = set()

    def __setitem__(self, key: str, data) -> None:
        if key in self.keys:
            return

        self.keys.add(key)
        kwargs = {
            "mode": "w",
            "force_zip64": True,
        }
        if self.zip is None or self.zip.fp is None:
            self.zip = zipfile.ZipFile(self.fn, mode="a", compression=zipfile.ZIP_DEFLATED)

        with self.zip.open(key, **kwargs) as fid:
            val = np.asanyarray(data)
            format.write_array(fid, val, allow_pickle=True)

    def __getitem__(self, key: str):
        self.zip.close()
        return np.load(self.fn, allow_pickle=True)[key]

    def __contains__(self, key: str):
        return key in self.keys

    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None

    def return_npz(self):
        self.zip.close()
        return np.load(self.fn, allow_pickle=True)
