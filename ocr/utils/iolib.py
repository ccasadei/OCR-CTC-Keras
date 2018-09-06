import os

from tensorflow.python.lib.io import file_io

import ocr


# Libreria di gestione file su filesystem locale e GCS
class IOLib:
    __tmpCnt = 0

    def __init__(self):
        self.__tmpFName = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __isGS__(self, fname):
        return fname.startswith("gs://")

    def __open__(self, fname, mode):
        if self.__isGS__(fname):
            return file_io.FileIO(fname, mode)
        else:
            return open(fname, mode)

    def delete(self, fname):
        if self.__isGS__(fname):
            file_io.delete_file(self.__tmpFName)
        else:
            os.remove(fname)

    def close(self):
        if self.__tmpFName is not None and self.fileExists(self.__tmpFName):
            self.delete(self.__tmpFName)

    def getTmpFileName(self):
        if self.__tmpFName is None:
            self.__tmpFName = os.path.join(os.path.dirname(ocr.__file__), "tmp_{:08d}.tmp".format(IOLib.__tmpCnt))
            IOLib.__tmpCnt += 1
        return self.__tmpFName

    def fileExists(self, fname):
        if self.__isGS__(fname):
            return file_io.file_exists(fname)
        else:
            return os.path.exists(fname)

    def copyToTmp(self, src):
        return self.copy(src, self.getTmpFileName())

    def copyFromTmp(self, dst):
        return self.copy(self.getTmpFileName(), dst)

    def copy(self, src, dst):
        if self.fileExists(src):
            with self.__open__(src, mode='rb') as input_f:
                with self.__open__(dst, mode='wb') as output_f:
                    output_f.write(input_f.read())
            return True
        else:
            return False
