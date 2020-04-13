import tempfile
import os

class TempRasterLayer:
    """Create a NamedTemporaryFile like object on Windows that has a close
    method

    Workaround used on Windows which cannot open the file a second time
    """

    def __init__(self):
        self.tfile = tempfile.NamedTemporaryFile().name
        self.name = self.tfile

    def close(self):
        os.unlink(self.tfile)


def _file_path_tempfile(file_path):
    """Returns a TemporaryFileWrapper and file path if a file_path parameter is None
    """
    if file_path is None:
        if os.name != "nt":
            tfile = tempfile.NamedTemporaryFile()
            file_path = tfile.name
        else:
            tfile = TempRasterLayer()
            file_path = tfile.name
    else:
        tfile = None

    return file_path, tfile
