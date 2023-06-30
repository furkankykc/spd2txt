# Created by H.Furkan KIYIKÇI at 01:07 21.03.2023 using PyCharm
# __author__ = 'Furkankykc'

import io
import struct
def decode(file_list):
    for i in range(1, len(file_list)):
        rf = file_list[i]
        with io.open(rf, 'rb') as rfh:
            wf = rf.replace(".spd", ".txt", 1)
            assert rf != wf
            with io.open(wf, 'w') as wfh:
                rfh.seek(1029)
                while True:
                    pair = rfh.read(16)
                    if not pair:
                        break
                    wl, A = struct.unpack("<dd", pair)
                    wfh.write(f"{wl}\t{A}\n")
