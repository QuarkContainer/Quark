// Copyright (c) 2021 Quark Container Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::SystemTime;
use std::ops::Deref;

use qobjs::common::*;
use qobjs::object_mgr::*;
use qobjs::object_mgr::SqlObjectMgr;
use qobjs::zip::ZipMgr;

#[derive(Debug)]
pub struct FuncDirMgrInner {
    pub baseDir: String,
    // cache of set of objName
    pub objNameCache: HashMap<String, SystemTime>,
    // mapping between last access time to the objName
    pub cacheTime: BTreeMap<SystemTime, String>,

    pub objectMgr: Arc<SqlObjectMgr>,
}

#[derive(Clone)]
pub struct FuncDirMgr(pub Arc<Mutex<FuncDirMgrInner>>);

impl Deref for FuncDirMgr {
    type Target = Arc<Mutex<FuncDirMgrInner>>;

    fn deref(&self) -> &Arc<Mutex<FuncDirMgrInner>> {
        &self.0
    }
}

pub const MAX_CACHE_DIR_COUNT : usize = 100;

impl FuncDirMgr {
    pub async fn New(baseDir: &str, objectDbAddr: &str) -> Result<Self> {
        let objectMgr = SqlObjectMgr::New(objectDbAddr).await?;
        std::fs::remove_dir_all(baseDir)?;
        let inner = FuncDirMgrInner {
            baseDir: baseDir.to_owned(),
            objectMgr: Arc::new(objectMgr),
            cacheTime: BTreeMap::new(),
            objNameCache: HashMap::new(),
        };

        return Ok(Self(Arc::new(Mutex::new(inner))));
    }

    pub fn DirName(&self, objName: &str) -> String {
        return format!("{}/{}", &self.lock().unwrap().baseDir, objName)
    }

    pub fn EvicateFuncDir(&self) -> Result<()> {
        let mut inner = self.lock().unwrap();
        if inner.objNameCache.len() <= MAX_CACHE_DIR_COUNT {
            return Ok(())
        }

        let count = inner.objNameCache.len() - MAX_CACHE_DIR_COUNT;
        for _i in 0..count {
            let (_time, name) = match inner.cacheTime.pop_first() {
                None => return Ok(()),
                Some(v) => v
            };

            inner.objNameCache.remove(&name);
            let folderName = format!("{}/{}", &inner.baseDir, name);
            std::fs::remove_dir_all(&folderName)?;
        }

        return Ok(())
    }

    pub async fn GetFuncDir(&self, namespace: &str, objName: &str) -> Result<String> {
        self.EvicateFuncDir()?;
        let objMgr = {
            let mut inner = self.lock().unwrap();
            match inner.objNameCache.get(objName).cloned() {
                None => (),
                Some(time) => {
                    let now= SystemTime::now();
                    
                    inner.objNameCache.insert(objName.to_owned(), now);
                    inner.cacheTime.remove(&time); 
                    inner.cacheTime.insert(now, objName.to_owned());
                    return Ok(self.DirName(objName))
                }
            }

            inner.objectMgr.clone()
        };
        
        let data = objMgr.ReadObject(namespace, objName).await?;
        let targetFolder = self.DirName(objName);
        let tmpFolder = self.DirName(objName) + ".tmp";
        
        std::fs::create_dir_all(&tmpFolder)?;
        ZipMgr::Unzip(&tmpFolder, data)?;

        let mut inner = self.lock().unwrap();
        match inner.objNameCache.get(objName) {
            None => (),
            Some(_time) => {
                std::fs::remove_dir_all(&tmpFolder)?;
                return Ok(self.DirName(objName))
            }
        }

        std::fs::rename(tmpFolder, targetFolder)?;
        let now = SystemTime::now();
        inner.cacheTime.insert(now, objName.to_owned());
        inner.objNameCache.insert(objName.to_owned(), now);

        return Ok(self.DirName(objName));
    }
}