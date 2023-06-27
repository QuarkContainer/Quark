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
from flask import Flask, render_template, request

app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(host='localhost',
                            database='auditdb',
                            user='audit_user',
                            password='123456')
    return conn


@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    namespace = request.args.get('namespace')
    query = "SELECT * FROM FuncAudit where namespace='{}';".format(namespace)
    cur.execute(query)
    audits = cur.fetchall()
    #print(audits)
    cur.close()
    conn.close()
    return render_template('index.html', audits=audits)

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=1234)