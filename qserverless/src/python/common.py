# Copyright (c) 2021 Quark Container Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy

from enum import Enum

class QErr:
    def __init__(self, err: str):
        self.err = err
    def __str__(self):
        return f"QErr: error is '{self.err}'"

class CallResult:
    def __init__(self, res: str, error: str):
        self.res = res
        self.error = error
    def __str__(self):
        return f"CallResult: res is '{self.res}', error is '{self.error}'"

class BlobAddr(dict):
    def __init__(self, blobSvcAddr: str, name: str):
        dict.__init__(self, blobSvcAddr=blobSvcAddr, name=name)
    
    def Set(self, addr):
        self['blobSvcAddr'] = addr['blobSvcAddr']
        self['name'] = addr['name']
    
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    @staticmethod
    def from_json(json_dct):
      return BlobAddr(json_dct['blobSvcAddr'], json_dct['name'])
  
    def __str__(self):
        return f"BlobAddr: svcAddr is '{self.blobSvcAddr}', name is '{self.name}'"

BlobAddrVec = list[BlobAddr]
BlobAddrMatrix = list[BlobAddrVec]

def TransposeBlobMatrix(mat: BlobAddrMatrix) -> BlobAddrMatrix:
    rows = len(mat)
    cols = len(mat[0])
    ret = list()
    for c in range(0, cols):
        row = list()
        for r in range(0, rows):
            row.append(mat[r][c])
        ret.append(row)
    return ret

def NewBlobAddrVec(colCount: int) -> BlobAddrVec:
    cols = list()
    for i in range(0, colCount):
        cols.append(BlobAddr(None, None))
    return cols

def NewBlobAddrMatrix(rowCount: int, colCount: int) -> BlobAddrVec:
    rows = list()
    for i in range(0, rowCount):
        rows.append(NewBlobAddrVec(colCount))
    return rows


# class BlobAddrVec(dict):
#     def __init__(self, colCount: int):
#         cols = list()
#         for i in range(0, colCount):
#             cols.append(BlobAddr(None, None))
#         dict.__init__(self, cols = cols)
    
#     def ColCount(self):
#         return len(self.vec)

# class BlobAddrMatrix:
#     def __init__(self, rows: int, cols: int):
#         self.rows = list()
#         for i in range(0, rows):
#             self.rows.append(BlobAddrVec(cols))
        
#     def Set(self, row: int, col: int, addr: BlobAddr):
#         self.rows[row][col] = addr
        
#     def Transpose(self):
#         newMat = BlobAddrMatrix(self.cols, self.rows)
#         newMat.rows = self.mat.transpose()
#         return newMat
    
#     def Row(self, idx: int) -> BlobAddrVec:
#         return self.rows[idx]
    
#     def toJson(self):
#         return json.dumps(self, default=lambda o: o.__dict__)
    
#     @staticmethod
#     def from_json(json_dct):
#         rows = json_dct['rows']
#         mat = BlobAddrMatrix(len(rows), len(rows[0]))
#         mat.rows = rows
#         return mat