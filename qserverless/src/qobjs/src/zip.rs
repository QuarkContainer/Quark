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

use std::path::Path;
use std::iter::Iterator;
use std::fs::File;
use std::io::Write;
use std::io::Read;

use zip::write::FileOptions;
use walkdir::WalkDir;

use crate::common::*;

pub struct ZipMgr {}

impl ZipMgr {
    pub fn ZipFolder(srcDir: &str) -> Result<Vec<u8>> {
        let method = zip::CompressionMethod::DEFLATE;
        if !Path::new(srcDir).is_dir() {
            return Err(Error::ENOENT(format!("zipfolder can't find folder {}", srcDir)));
        }

        let walkdir = WalkDir::new(srcDir);
        let it = walkdir.into_iter();
        let it = &mut it.filter_map(|e| e.ok());

        let mut zip_data: Vec<u8> = Vec::new();
        {
            // Create a ZipWriter with the Vec<u8> buffer
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut zip_data));
            let options = FileOptions::default()
                .compression_method(method)
                .unix_permissions(0o755);

            let mut buffer = Vec::new();
            for entry in it {
                let path = entry.path();
                let name = path.strip_prefix(Path::new(srcDir)).unwrap();
        
                // Write file or directory explicitly
                // Some unzip tools unzip files with directory paths correctly, some do not!
                if path.is_file() {
                    println!("adding file {path:?} as {name:?} ...");
                    #[allow(deprecated)]
                    zip.start_file_from_path(name, options)?;
                    let mut f = File::open(path)?;
        
                    f.read_to_end(&mut buffer)?;
                    zip.write_all(&buffer)?;
                    buffer.clear();
                } else if !name.as_os_str().is_empty() {
                    // Only if not root! Avoids path spec / warning
                    // and mapname conversion failed error on unzip
                    println!("adding dir {path:?} as {name:?} ...");
                    #[allow(deprecated)]
                    zip.add_directory_from_path(name, options)?;
                }
            }
            zip.finish()?;
        }
        
        return Ok(zip_data)
    }

    pub fn Unzip(folder: &str, data: Vec<u8>) -> Result<()> {
        let filename = format!("{}/{}", folder, "archive.zip");
        std::fs::write(&filename, &data)?;
        let output = std::process::Command::new("unzip")
            .arg(&filename)  // Specify the archive file
            .arg("-d")           // Specify the destination directory
            .arg(folder)   // Specify the output directory
            .output()?;

            // Check the exit status of the unzip command
        if output.status.success() {
            return Ok(())
        } else {
            let error_message = String::from_utf8_lossy(&output.stderr);
            return Err(Error::CommonError(format!("Unzip operation failed: {}", error_message)));
        }
    }
}