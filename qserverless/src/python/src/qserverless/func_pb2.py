# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: func.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nfunc.proto\x12\x04\x66unc\"\x85\x02\n\nBlobSvcReq\x12\r\n\x05msgId\x18\x01 \x01(\x04\x12)\n\x0b\x42lobOpenReq\x18\xf5\x03 \x01(\x0b\x32\x11.func.BlobOpenReqH\x00\x12)\n\x0b\x42lobReadReq\x18\xf9\x03 \x01(\x0b\x32\x11.func.BlobReadReqH\x00\x12)\n\x0b\x42lobSeekReq\x18\xfb\x03 \x01(\x0b\x32\x11.func.BlobSeekReqH\x00\x12+\n\x0c\x42lobCloseReq\x18\x81\x04 \x01(\x0b\x32\x12.func.BlobCloseReqH\x00\x12-\n\rBlobDeleteReq\x18\x83\x04 \x01(\x0b\x32\x13.func.BlobDeleteReqH\x00\x42\x0b\n\tEventBody\"\x90\x02\n\x0b\x42lobSvcResp\x12\r\n\x05msgId\x18\x01 \x01(\x04\x12+\n\x0c\x42lobOpenResp\x18\xf6\x03 \x01(\x0b\x32\x12.func.BlobOpenRespH\x00\x12+\n\x0c\x42lobReadResp\x18\xfa\x03 \x01(\x0b\x32\x12.func.BlobReadRespH\x00\x12+\n\x0c\x42lobSeekResp\x18\xfc\x03 \x01(\x0b\x32\x12.func.BlobSeekRespH\x00\x12-\n\rBlobCloseResp\x18\x82\x04 \x01(\x0b\x32\x13.func.BlobCloseRespH\x00\x12/\n\x0e\x42lobDeleteResp\x18\x84\x04 \x01(\x0b\x32\x14.func.BlobDeleteRespH\x00\x42\x0b\n\tEventBody\"\xe1\x07\n\x0c\x46uncAgentMsg\x12\r\n\x05msgId\x18\x01 \x01(\x04\x12\x36\n\x12\x46uncPodRegisterReq\x18\x64 \x01(\x0b\x32\x18.func.FuncPodRegisterReqH\x00\x12\x39\n\x13\x46uncPodRegisterResp\x18\xc8\x01 \x01(\x0b\x32\x19.func.FuncPodRegisterRespH\x00\x12\x33\n\x10\x46uncAgentCallReq\x18\xac\x02 \x01(\x0b\x32\x16.func.FuncAgentCallReqH\x00\x12\x35\n\x11\x46uncAgentCallResp\x18\x90\x03 \x01(\x0b\x32\x17.func.FuncAgentCallRespH\x00\x12\x33\n\x10\x46uncAgentCallAck\x18\x91\x03 \x01(\x0b\x32\x16.func.FuncAgentCallAckH\x00\x12)\n\x0b\x42lobOpenReq\x18\xf5\x03 \x01(\x0b\x32\x11.func.BlobOpenReqH\x00\x12+\n\x0c\x42lobOpenResp\x18\xf6\x03 \x01(\x0b\x32\x12.func.BlobOpenRespH\x00\x12-\n\rBlobCreateReq\x18\xf7\x03 \x01(\x0b\x32\x13.func.BlobCreateReqH\x00\x12/\n\x0e\x42lobCreateResp\x18\xf8\x03 \x01(\x0b\x32\x14.func.BlobCreateRespH\x00\x12)\n\x0b\x42lobReadReq\x18\xf9\x03 \x01(\x0b\x32\x11.func.BlobReadReqH\x00\x12+\n\x0c\x42lobReadResp\x18\xfa\x03 \x01(\x0b\x32\x12.func.BlobReadRespH\x00\x12)\n\x0b\x42lobSeekReq\x18\xfb\x03 \x01(\x0b\x32\x11.func.BlobSeekReqH\x00\x12+\n\x0c\x42lobSeekResp\x18\xfc\x03 \x01(\x0b\x32\x12.func.BlobSeekRespH\x00\x12+\n\x0c\x42lobWriteReq\x18\xfd\x03 \x01(\x0b\x32\x12.func.BlobWriteReqH\x00\x12-\n\rBlobWriteResp\x18\xfe\x03 \x01(\x0b\x32\x13.func.BlobWriteRespH\x00\x12+\n\x0c\x42lobCloseReq\x18\x81\x04 \x01(\x0b\x32\x12.func.BlobCloseReqH\x00\x12-\n\rBlobCloseResp\x18\x82\x04 \x01(\x0b\x32\x13.func.BlobCloseRespH\x00\x12-\n\rBlobDeleteReq\x18\x83\x04 \x01(\x0b\x32\x13.func.BlobDeleteReqH\x00\x12/\n\x0e\x42lobDeleteResp\x18\x84\x04 \x01(\x0b\x32\x14.func.BlobDeleteRespH\x00\x12!\n\x07\x46uncMsg\x18\x85\x04 \x01(\x0b\x32\r.func.FuncMsgH\x00\x42\x0b\n\tEventBody\"?\n\x0b\x42lobOpenReq\x12\x0f\n\x07svcAddr\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\"\xb8\x01\n\x0c\x42lobOpenResp\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\x12\x0c\n\x04size\x18\x05 \x01(\x04\x12\x10\n\x08\x63hecksum\x18\x06 \x01(\t\x12#\n\ncreateTime\x18\x07 \x01(\x0b\x32\x0f.func.Timestamp\x12\'\n\x0elastAccessTime\x18\x08 \x01(\x0b\x32\x0f.func.Timestamp\x12\r\n\x05\x65rror\x18\t \x01(\t\"A\n\rBlobDeleteReq\x12\x0f\n\x07svcAddr\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\"\x1f\n\x0e\x42lobDeleteResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"0\n\rBlobCreateReq\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\"<\n\x0e\x42lobCreateResp\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0f\n\x07svcAddr\x18\x03 \x01(\t\x12\r\n\x05\x65rror\x18\t \x01(\t\"&\n\x0b\x42lobReadReq\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0b\n\x03len\x18\x03 \x01(\x04\"+\n\x0c\x42lobReadResp\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\r\n\x05\x65rror\x18\x04 \x01(\t\"8\n\x0b\x42lobSeekReq\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0b\n\x03pos\x18\x03 \x01(\x03\x12\x10\n\x08seekType\x18\x04 \x01(\r\"-\n\x0c\x42lobSeekResp\x12\x0e\n\x06offset\x18\x02 \x01(\x04\x12\r\n\x05\x65rror\x18\x03 \x01(\t\"\x1a\n\x0c\x42lobCloseReq\x12\n\n\x02id\x18\x02 \x01(\x04\"\x1e\n\rBlobCloseResp\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"(\n\x0c\x42lobWriteReq\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"\x1e\n\rBlobWriteResp\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"\x19\n\x0b\x42lobSealReq\x12\n\n\x02id\x18\x02 \x01(\x04\"\x1d\n\x0c\x42lobSealResp\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"c\n\x12\x46uncPodRegisterReq\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\x12\x11\n\tnamespace\x18\x02 \x01(\t\x12\x13\n\x0bpackageName\x18\x03 \x01(\t\x12\x12\n\nclientMode\x18\x04 \x01(\x08\"$\n\x13\x46uncPodRegisterResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"\xe0\x01\n\x10\x46uncAgentCallReq\x12\r\n\x05jobId\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x13\n\x0bpackageName\x18\x04 \x01(\t\x12\x10\n\x08\x66uncName\x18\x05 \x01(\t\x12\x12\n\nparameters\x18\x06 \x01(\t\x12\x10\n\x08priority\x18\x07 \x01(\x04\x12\x14\n\x0c\x63\x61llerFuncId\x18\x08 \x01(\t\x12\x14\n\x0c\x63\x61llerNodeId\x18\t \x01(\t\x12\x13\n\x0b\x63\x61llerPodId\x18\n \x01(\t\x12\x10\n\x08\x63\x61llType\x18\x0b \x01(\x05\";\n\x11\x46uncAgentCallResp\x12\n\n\x02id\x18\x01 \x01(\t\x12\x1a\n\x03res\x18\x02 \x01(\x0b\x32\r.func.FuncRes\"\x83\x01\n\x10\x46uncAgentCallAck\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x14\n\x0c\x63\x61lleeNodeId\x18\x03 \x01(\t\x12\x13\n\x0b\x63\x61lleePodId\x18\x04 \x01(\t\x12\x14\n\x0c\x63\x61llerNodeId\x18\x08 \x01(\t\x12\x13\n\x0b\x63\x61llerPodId\x18\t \x01(\t\"\x1e\n\x02KV\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x0b\n\x03val\x18\x02 \x01(\t\"\xe5\x01\n\x07\x46uncMsg\x12\r\n\x05msgId\x18\x01 \x01(\t\x12\x11\n\tsrcNodeId\x18\x02 \x01(\t\x12\x10\n\x08srcPodId\x18\x03 \x01(\t\x12\x11\n\tsrcFuncId\x18\x04 \x01(\t\x12\x11\n\tdstNodeId\x18\x05 \x01(\t\x12\x10\n\x08\x64stPodId\x18\x06 \x01(\t\x12\x11\n\tdstFuncId\x18\x07 \x01(\t\x12(\n\x0b\x46uncMsgBody\x18\x65 \x01(\x0b\x32\x11.func.FuncMsgBodyH\x00\x12&\n\nFuncMsgAck\x18\x66 \x01(\x0b\x32\x10.func.FuncMsgAckH\x00\x42\t\n\x07Payload\"\x1b\n\x0b\x46uncMsgBody\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\"\x1b\n\nFuncMsgAck\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"\xa0\x04\n\nFuncSvcMsg\x12:\n\x14\x46uncAgentRegisterReq\x18\x64 \x01(\x0b\x32\x1a.func.FuncAgentRegisterReqH\x00\x12=\n\x15\x46uncAgentRegisterResp\x18\xc8\x01 \x01(\x0b\x32\x1b.func.FuncAgentRegisterRespH\x00\x12/\n\x0e\x46uncPodConnReq\x18\xac\x02 \x01(\x0b\x32\x14.func.FuncPodConnReqH\x00\x12\x31\n\x0f\x46uncPodConnResp\x18\x90\x03 \x01(\x0b\x32\x15.func.FuncPodConnRespH\x00\x12\x35\n\x11\x46uncPodDisconnReq\x18\xf4\x03 \x01(\x0b\x32\x17.func.FuncPodDisconnReqH\x00\x12\x37\n\x12\x46uncPodDisconnResp\x18\xd8\x04 \x01(\x0b\x32\x18.func.FuncPodDisconnRespH\x00\x12/\n\x0e\x46uncSvcCallReq\x18\xbc\x05 \x01(\x0b\x32\x14.func.FuncSvcCallReqH\x00\x12\x31\n\x0f\x46uncSvcCallResp\x18\xa0\x06 \x01(\x0b\x32\x15.func.FuncSvcCallRespH\x00\x12/\n\x0e\x46uncSvcCallAck\x18\xa1\x06 \x01(\x0b\x32\x14.func.FuncSvcCallAckH\x00\x12!\n\x07\x46uncMsg\x18\x84\x07 \x01(\x0b\x32\r.func.FuncMsgH\x00\x42\x0b\n\tEventBody\"\xc5\x01\n\x14\x46uncAgentRegisterReq\x12\x0e\n\x06nodeId\x18\x01 \x01(\t\x12)\n\x0b\x63\x61llerCalls\x18\x02 \x03(\x0b\x32\x14.func.FuncSvcCallReq\x12)\n\x0b\x63\x61lleeCalls\x18\x03 \x03(\x0b\x32\x14.func.FuncSvcCallReq\x12%\n\x08\x66uncPods\x18\x04 \x03(\x0b\x32\x13.func.FuncPodStatus\x12 \n\x08resource\x18\x05 \x01(\x0b\x32\x0e.func.Resource\"$\n\x08Resource\x12\x0b\n\x03mem\x18\x01 \x01(\x04\x12\x0b\n\x03\x63pu\x18\x02 \x01(\r\"&\n\x15\x46uncAgentRegisterResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"_\n\x0e\x46uncPodConnReq\x12\x11\n\tfuncPodId\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x13\n\x0bpackageName\x18\x04 \x01(\t\x12\x12\n\nclientMode\x18\x05 \x01(\x08\"3\n\x0f\x46uncPodConnResp\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"&\n\x11\x46uncPodDisconnReq\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\"#\n\x12\x46uncPodDisconnResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"\xae\x02\n\x0e\x46uncSvcCallReq\x12\r\n\x05jobId\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x13\n\x0bpackageName\x18\x04 \x01(\t\x12\x10\n\x08\x66uncName\x18\x05 \x01(\t\x12\x12\n\nparameters\x18\x06 \x01(\t\x12\x10\n\x08priority\x18\x07 \x01(\x04\x12#\n\ncreatetime\x18\t \x01(\x0b\x32\x0f.func.Timestamp\x12\x14\n\x0c\x63\x61llerNodeId\x18\n \x01(\t\x12\x13\n\x0b\x63\x61llerPodId\x18\x0b \x01(\t\x12\x14\n\x0c\x63\x61llerFuncId\x18\x08 \x01(\t\x12\x14\n\x0c\x63\x61lleeNodeId\x18\x0c \x01(\t\x12\x13\n\x0b\x63\x61lleePodId\x18\r \x01(\t\x12\x10\n\x08\x63\x61llType\x18\x0e \x01(\x05\"\x8f\x01\n\x0f\x46uncSvcCallResp\x12\n\n\x02id\x18\x01 \x01(\t\x12\x1a\n\x03res\x18\x02 \x01(\x0b\x32\r.func.FuncRes\x12\x14\n\x0c\x63\x61llerNodeId\x18\x08 \x01(\t\x12\x13\n\x0b\x63\x61llerPodId\x18\t \x01(\t\x12\x14\n\x0c\x63\x61lleeNodeId\x18\n \x01(\t\x12\x13\n\x0b\x63\x61lleePodId\x18\x0b \x01(\t\"&\n\x05\x45rror\x12\x0e\n\x06source\x18\x01 \x01(\x05\x12\r\n\x05\x65rror\x18\x02 \x01(\t\">\n\x07\x46uncRes\x12\x1c\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x0b.func.ErrorH\x00\x12\x0e\n\x04resp\x18\x03 \x01(\tH\x00\x42\x05\n\x03res\"\x81\x01\n\x0e\x46uncSvcCallAck\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x14\n\x0c\x63\x61lleeNodeId\x18\x03 \x01(\t\x12\x13\n\x0b\x63\x61lleePodId\x18\x04 \x01(\t\x12\x14\n\x0c\x63\x61llerNodeId\x18\x08 \x01(\t\x12\x13\n\x0b\x63\x61llerPodId\x18\t \x01(\t\"\x95\x01\n\rFuncPodStatus\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\x12\x11\n\tnamespace\x18\x02 \x01(\t\x12\x13\n\x0bpackageName\x18\x03 \x01(\t\x12!\n\x05state\x18\x04 \x01(\x0e\x32\x12.func.FuncPodState\x12\x12\n\nfuncCallId\x18\x05 \x01(\t\x12\x12\n\nclientMode\x18\x06 \x01(\x08\"+\n\tTimestamp\x12\x0f\n\x07seconds\x18\x01 \x01(\x04\x12\r\n\x05nanos\x18\x02 \x01(\r*%\n\x0c\x46uncPodState\x12\x08\n\x04Idle\x10\x00\x12\x0b\n\x07Running\x10\x01\x32G\n\x0b\x42lobService\x12\x38\n\rStreamProcess\x12\x10.func.BlobSvcReq\x1a\x11.func.BlobSvcResp(\x01\x30\x01\x32\x8c\x01\n\x10\x46uncAgentService\x12;\n\rStreamProcess\x12\x12.func.FuncAgentMsg\x1a\x12.func.FuncAgentMsg(\x01\x30\x01\x12;\n\x08\x46uncCall\x12\x16.func.FuncAgentCallReq\x1a\x17.func.FuncAgentCallResp2I\n\x0e\x46uncSvcService\x12\x37\n\rStreamProcess\x12\x10.func.FuncSvcMsg\x1a\x10.func.FuncSvcMsg(\x01\x30\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'func_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FUNCPODSTATE._serialized_start=5220
  _FUNCPODSTATE._serialized_end=5257
  _BLOBSVCREQ._serialized_start=21
  _BLOBSVCREQ._serialized_end=282
  _BLOBSVCRESP._serialized_start=285
  _BLOBSVCRESP._serialized_end=557
  _FUNCAGENTMSG._serialized_start=560
  _FUNCAGENTMSG._serialized_end=1553
  _BLOBOPENREQ._serialized_start=1555
  _BLOBOPENREQ._serialized_end=1618
  _BLOBOPENRESP._serialized_start=1621
  _BLOBOPENRESP._serialized_end=1805
  _BLOBDELETEREQ._serialized_start=1807
  _BLOBDELETEREQ._serialized_end=1872
  _BLOBDELETERESP._serialized_start=1874
  _BLOBDELETERESP._serialized_end=1905
  _BLOBCREATEREQ._serialized_start=1907
  _BLOBCREATEREQ._serialized_end=1955
  _BLOBCREATERESP._serialized_start=1957
  _BLOBCREATERESP._serialized_end=2017
  _BLOBREADREQ._serialized_start=2019
  _BLOBREADREQ._serialized_end=2057
  _BLOBREADRESP._serialized_start=2059
  _BLOBREADRESP._serialized_end=2102
  _BLOBSEEKREQ._serialized_start=2104
  _BLOBSEEKREQ._serialized_end=2160
  _BLOBSEEKRESP._serialized_start=2162
  _BLOBSEEKRESP._serialized_end=2207
  _BLOBCLOSEREQ._serialized_start=2209
  _BLOBCLOSEREQ._serialized_end=2235
  _BLOBCLOSERESP._serialized_start=2237
  _BLOBCLOSERESP._serialized_end=2267
  _BLOBWRITEREQ._serialized_start=2269
  _BLOBWRITEREQ._serialized_end=2309
  _BLOBWRITERESP._serialized_start=2311
  _BLOBWRITERESP._serialized_end=2341
  _BLOBSEALREQ._serialized_start=2343
  _BLOBSEALREQ._serialized_end=2368
  _BLOBSEALRESP._serialized_start=2370
  _BLOBSEALRESP._serialized_end=2399
  _FUNCPODREGISTERREQ._serialized_start=2401
  _FUNCPODREGISTERREQ._serialized_end=2500
  _FUNCPODREGISTERRESP._serialized_start=2502
  _FUNCPODREGISTERRESP._serialized_end=2538
  _FUNCAGENTCALLREQ._serialized_start=2541
  _FUNCAGENTCALLREQ._serialized_end=2765
  _FUNCAGENTCALLRESP._serialized_start=2767
  _FUNCAGENTCALLRESP._serialized_end=2826
  _FUNCAGENTCALLACK._serialized_start=2829
  _FUNCAGENTCALLACK._serialized_end=2960
  _KV._serialized_start=2962
  _KV._serialized_end=2992
  _FUNCMSG._serialized_start=2995
  _FUNCMSG._serialized_end=3224
  _FUNCMSGBODY._serialized_start=3226
  _FUNCMSGBODY._serialized_end=3253
  _FUNCMSGACK._serialized_start=3255
  _FUNCMSGACK._serialized_end=3282
  _FUNCSVCMSG._serialized_start=3285
  _FUNCSVCMSG._serialized_end=3829
  _FUNCAGENTREGISTERREQ._serialized_start=3832
  _FUNCAGENTREGISTERREQ._serialized_end=4029
  _RESOURCE._serialized_start=4031
  _RESOURCE._serialized_end=4067
  _FUNCAGENTREGISTERRESP._serialized_start=4069
  _FUNCAGENTREGISTERRESP._serialized_end=4107
  _FUNCPODCONNREQ._serialized_start=4109
  _FUNCPODCONNREQ._serialized_end=4204
  _FUNCPODCONNRESP._serialized_start=4206
  _FUNCPODCONNRESP._serialized_end=4257
  _FUNCPODDISCONNREQ._serialized_start=4259
  _FUNCPODDISCONNREQ._serialized_end=4297
  _FUNCPODDISCONNRESP._serialized_start=4299
  _FUNCPODDISCONNRESP._serialized_end=4334
  _FUNCSVCCALLREQ._serialized_start=4337
  _FUNCSVCCALLREQ._serialized_end=4639
  _FUNCSVCCALLRESP._serialized_start=4642
  _FUNCSVCCALLRESP._serialized_end=4785
  _ERROR._serialized_start=4787
  _ERROR._serialized_end=4825
  _FUNCRES._serialized_start=4827
  _FUNCRES._serialized_end=4889
  _FUNCSVCCALLACK._serialized_start=4892
  _FUNCSVCCALLACK._serialized_end=5021
  _FUNCPODSTATUS._serialized_start=5024
  _FUNCPODSTATUS._serialized_end=5173
  _TIMESTAMP._serialized_start=5175
  _TIMESTAMP._serialized_end=5218
  _BLOBSERVICE._serialized_start=5259
  _BLOBSERVICE._serialized_end=5330
  _FUNCAGENTSERVICE._serialized_start=5333
  _FUNCAGENTSERVICE._serialized_end=5473
  _FUNCSVCSERVICE._serialized_start=5475
  _FUNCSVCSERVICE._serialized_end=5548
# @@protoc_insertion_point(module_scope)
