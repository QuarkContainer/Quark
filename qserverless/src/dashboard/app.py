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

import os
import psycopg2
from flask import Flask, render_template, request, render_template_string
import grpc
import json

import qobjs_pb2
import qobjs_pb2_grpc

app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(host='localhost',
                            database='auditdb',
                            user='audit_user',
                            password='123456')
    return conn

def ReadFuncLog(namespace: str, funcId: str) -> str: 
    req = qobjs_pb2.ReadFuncLogReq (
        namespace = namespace,
        funcName = funcId,
    )
    
    channel = grpc.insecure_channel("127.0.0.1:8890")
    stub = qobjs_pb2_grpc.QMetaServiceStub(channel)
    res = stub.ReadFuncLog(req)
    return res.content

def listpackages(namespace: str): 
    req = qobjs_pb2.GetRequestMessage (
        obj_type = "package",
        namespace = namespace,
    )
    
    channel = grpc.insecure_channel("127.0.0.1:8890")
    stub = qobjs_pb2_grpc.QMetaServiceStub(channel)
    res = stub.List(req)
    
    packages = []
    for obj in res.objs:
        data = json.loads(obj.data) 
        jsonstr = json.dumps(data, indent = 4)
        jsonstr = jsonstr.replace('\n', '<br>')
        jsonstr = jsonstr.replace('    ', '&emsp;')
        packages.append((obj.namespace, obj.name, jsonstr, obj.revision))
    
    return packages

def getpackage(namespace: str, pacakgeName: str):
    req = qobjs_pb2.GetRequestMessage (
        obj_type = "package",
        namespace = namespace,
        name = pacakgeName
    )     
    channel = grpc.insecure_channel("127.0.0.1:8890")
    stub = qobjs_pb2_grpc.QMetaServiceStub(channel)
    res = stub.Get(req)
    obj = res.obj
    data = json.loads(obj.data) 
    jsonstr = json.dumps(data, indent = 4)
    jsonstr = jsonstr.replace('\n', '<br>')
    jsonstr = jsonstr.replace('    ', '&emsp;')
    return (jsonstr, obj.revision)

@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    namespace = request.args.get('namespace')
    jobid = request.args.get('jobid')
    query = "SELECT * FROM FuncAudit where namespace='{}' order by createTime desc;".format(namespace)
    if jobid is not None:
        query = "SELECT * FROM FuncAudit where namespace='{}' and jobId = '{}' order by createTime desc;".format(namespace, jobid)
    
    cur.execute(query)
    audits = cur.fetchall()
    
    cur.close()
    conn.close()
    hosturl = request.host_url
    return render_template('index.html', audits=audits, hosturl=hosturl, namespace=namespace)

@app.route('/funclog')
def funclog():
    namespace = request.args.get('namespace')
    #funcId = "d86ac7ba-a1c0-4f45-9215-af6fe6b1d7bb"
    funcId = request.args.get('funcId')
    funcName = request.args.get('funcName')
    log = ReadFuncLog(namespace, funcId)
    output = log.replace('\n', '<br>')
    return render_template('log.html', namespace=namespace, funcId=funcId, funcName=funcName, log=output)

@app.route('/listpackage')
def ListPackage():
    namespace = request.args.get('namespace')
    packages = listpackages(namespace)
    return render_template('package_list.html', packages=packages)
    
@app.route('/package')
def GetPackage():
    namespace = request.args.get('namespace')
    name = request.args.get('name')
    (package, revision) = getpackage(namespace, name)
    return render_template('package.html', namespace=namespace, name=name, package=package, revision=revision)
    


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=1234)