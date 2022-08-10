// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::common::*;

pub struct Lazybuf<'a> {
    s: &'a [u8],
    buf: Option<Vec<u8>>,
    w: usize,
}

impl<'a> Lazybuf<'a> {
    pub fn New(s: &'a str) -> Self {
        return Self {
            s: s.as_bytes(),
            buf: None,
            w: 0,
        };
    }

    pub fn index(&self, i: usize) -> u8 {
        match &self.buf {
            Some(ref b) => return b[i],
            None => return self.s[i],
        }
    }

    pub fn append(&mut self, c: u8) {
        let alloc = match self.buf {
            None => {
                if self.w < self.s.len() && self.s[self.w] == c {
                    self.w += 1;
                    return;
                }

                false
            }

            _ => true,
        };

        if !alloc {
            let mut buf = Vec::with_capacity(self.s.len());
            for _i in 0..self.s.len() {
                buf.push(0);
            }
            buf.copy_from_slice(&self.s);
            self.buf = Some(buf)
        }

        let buf = self.buf.as_deref_mut().unwrap();
        buf[self.w] = c;
        self.w += 1;
    }

    pub fn String(&self) -> String {
        match &self.buf {
            None => return String::from_utf8(self.s[..self.w].to_vec()).unwrap(),
            Some(b) => return String::from_utf8(b[..self.w].to_vec()).unwrap(),
        }
    }
}

pub fn Clean(path: &str) -> String {
    if path == "" {
        return ".".to_string();
    }

    let mut out = Lazybuf::New(path);
    let path = path.as_bytes();
    let n = path.len();

    let rooted = path[0] == '/' as u8;
    let mut r = 0;
    let mut dotdot = 0;

    if rooted {
        out.append('/' as u8);
        r = 1;
        dotdot = 1;
    }

    while r < n {
        if path[r] == '/' as u8 {
            r += 1;
        } else if path[r] == '.' as u8 && (r + 1 == n || path[r + 1] == '/' as u8) {
            r += 1;
        } else if path[r] == '.' as u8
            && path[r + 1] == '.' as u8
            && (r + 2 == n || path[r + 2] == '/' as u8)
        {
            r += 2;
            if out.w > dotdot {
                out.w -= 1;
                while out.w > dotdot && out.index(out.w) != '/' as u8 {
                    out.w -= 1;
                }
            } else if !rooted {
                if out.w > 0 {
                    out.append('/' as u8);
                }
                out.append('.' as u8);
                out.append('.' as u8);
                dotdot = out.w;
            }
        } else {
            if rooted && out.w != 1 || !rooted && out.w != 0 {
                out.append('/' as u8)
            }

            while r < n && path[r] != '/' as u8 {
                out.append(path[r]);
                r += 1;
            }
        }
    }

    if out.w == 0 {
        return ".".to_string();
    }

    return out.String();
}

pub fn Split<'a>(path: &'a str) -> (&'a str, &'a str) {
    let i = path.rfind('/');

    match i {
        None => ("", &path[..]),
        Some(i) => (&path[..i + 1], &path[i + 1..]),
    }
}

pub fn Join(path: &str, elem: &str) -> String {
    let res = path.to_string() + &"/" + elem;

    return Clean(&res);
}

pub fn Ext<'a>(path: &'a str) -> &'a str {
    let s = path.as_bytes();
    for i in (0..s.len()).rev() {
        if s[i] == '/' as u8 {
            break;
        }

        if s[i] == '.' as u8 {
            return &path[i..];
        }
    }

    return "";
}

pub fn Base<'a>(path: &'a str) -> &'a str {
    if path == "" {
        return ".";
    }

    let s = &path.as_bytes()[..];

    let mut r = s.len() - 1;

    while s[r] == '/' as u8 {
        if r > 0 {
            r -= 1;
        } else {
            return "/";
        }
    }

    let mut l = 0;
    for i in (0..r).rev() {
        if s[i] == '/' as u8 {
            l = i + 1;
            break;
        }
    }

    if l == r {
        return "/";
    }

    return &path[l..r + 1];
}

pub fn IsAbs(path: &str) -> bool {
    return path.len() > 0 && path.as_bytes()[0] == '/' as u8;
}

pub fn Dir(path: &str) -> String {
    let (dir, _) = Split(path);
    return Clean(&dir.to_string());
}

pub fn TrimTrailingSlashes<'a>(dir: &'a str) -> (&'a str, bool) {
    if dir.len() == 0 {
        return (dir, false);
    }

    let s = &dir.as_bytes()[..];
    let mut changed = false;

    let mut last = (dir.len() - 1) as isize;
    while last >= 1 {
        if s[last as usize] != '/' as u8 {
            break;
        }

        last -= 1;
        changed = true;
    }

    return (&dir[..(last + 1) as usize], changed);
}

pub fn LastIndex(path: &str, c: u8) -> i32 {
    for i in 0..path.len() {
        let idx = path.len() - i - 1;
        if path.as_bytes()[idx] == c {
            return idx as i32;
        }
    }

    return -1;
}

//return (dir, file)
pub fn SplitLast<'a>(path: &'a str) -> (&'a str, &'a str) {
    let (path, _) = TrimTrailingSlashes(path);

    if path == "" {
        return (&".", &".");
    } else if path == "/" {
        return (&"/", &".");
    }

    let mut slash = (path.len() - 1) as isize;
    let s = &path.as_bytes()[..];

    while slash >= 0 {
        if s[slash as usize] == '/' as u8 {
            break;
        }

        slash -= 1;
    }

    if slash < 0 {
        return (&".", path);
    } else if slash == 0 {
        return (&"/", &path[1..]);
    } else {
        let slash = slash as usize;
        let (dir, _) = TrimTrailingSlashes(&path[..slash]);
        return (dir, &path[slash + 1..]);
    }
}

//return (first, remain)
pub fn SplitFirst<'a>(path: &'a str) -> (&'a str, &'a str) {
    let (path, _) = TrimTrailingSlashes(path);
    if path == "" {
        return (&".", &"");
    }

    let mut slash = 0;
    let s = &path.as_bytes()[..];

    while slash < path.len() {
        if s[slash] == '/' as u8 {
            break;
        }

        slash += 1;
    }

    if slash >= path.len() {
        return (path, &"");
    } else if slash == 0 {
        return (&"/", &path[1..]);
    } else {
        let current = &path[..slash];
        let mut remain = &path[slash + 1..];
        let mut s = &remain.as_bytes()[..];

        while remain.len() > 0 && s[0] == '/' as u8 {
            remain = &remain[1..];
            s = &remain.as_bytes()[..];
        }

        return (current, remain);
    }
}

// IsSubpath checks whether the first path is a (strict) descendent of the
// second. If it is a subpath, then true is returned along with a clean
// relative path from the second path to the first. Otherwise false is
// returned.
pub fn IsSubpath(subpath: &str, path: &str) -> (String, bool) {
    let mut cleanPath = Clean(path);
    let cleanSubpath = Clean(subpath);

    let s = &cleanPath.as_bytes()[..];
    if cleanPath.len() == 0 || s[s.len() - 1] == '/' as u8 {
        cleanPath += "/";
    }

    if cleanPath == cleanSubpath {
        return ("".to_string(), false);
    }

    if HasPrefix(&cleanSubpath, &cleanPath) {
        return (TrimPrefix(&cleanSubpath, &cleanPath), true);
    }

    return ("".to_string(), false);
}

pub fn HasPrefix(s: &str, prefix: &str) -> bool {
    return s.len() >= prefix.len() && s[..prefix.len()] == prefix[..];
}

pub fn TrimPrefix(s: &str, prefix: &str) -> String {
    if HasPrefix(s, prefix) {
        return s[prefix.len()..].to_string();
    }

    return s.to_string();
}

fn volumeNameLen(_path: &str) -> usize {
    return 0;
}

pub fn VolumeName<'a>(path: &'a str) -> &'a str {
    return &path[..volumeNameLen(path)];
}

pub const PATH_SEPARATOR: char = '/';
pub const PATH_LIST_SEPARATOR: char = ':';

// Rel returns a relative path that is lexically equivalent to targpath when
// joined to basepath with an intervening separator. That is,
// Join(basepath, Rel(basepath, targpath)) is equivalent to targpath itself.
// On success, the returned path will always be relative to basepath,
// even if basepath and targpath share no elements.
// An error is returned if targpath can't be made relative to basepath or if
// knowing the current working directory would be necessary to compute it.
// Rel calls Clean on the result.
pub fn Rel(basepath: &str, targpath: &str) -> Result<String> {
    let PathSeparator = PATH_SEPARATOR;

    let baseVol = VolumeName(basepath);
    let targVol = VolumeName(targpath);

    let baseStr = Clean(&basepath.to_string());
    let targStr = Clean(&targpath.to_string());
    let mut base = baseStr.as_bytes();
    let mut targ = targStr.as_bytes();
    if targ == base {
        return Ok(".".to_string());
    }

    base = &base[baseVol.len()..];
    targ = &targ[baseVol.len()..];
    if base == ".".as_bytes() {
        base = "".as_bytes();
    }

    let baseSlashed = base.len() > 0 && base[0] == PathSeparator as u8;
    let targSlashed = targ.len() > 0 && targ[0] == PathSeparator as u8;

    if baseSlashed != targSlashed || baseVol != targVol {
        return Err(Error::Common("Rel: can't make".to_string()));
    }

    let bl = base.len();
    let tl = targ.len();

    let mut b0 = 0;
    let mut bi = 0;
    let mut t0 = 0;
    let mut ti = 0;

    loop {
        while bi < bl && base[bi] != PathSeparator as u8 {
            bi += 1;
        }

        while ti < tl && targ[ti] != PathSeparator as u8 {
            ti += 1;
        }

        if targ[t0..ti] != base[b0..bi] {
            break;
        }

        if bi < bl {
            bi += 1;
        }

        if ti < tl {
            ti += 1;
        }

        b0 = bi;
        t0 = ti;
    }

    if &base[b0..bi] == "..".as_bytes() {
        return Err(Error::Common("Rel: can't make".to_string()));
    }

    if b0 != bl {
        let mut seps = 0;
        for i in b0..bl + 1 {
            if base[i] == PathSeparator as u8 {
                seps += 1;
            }
        }

        let mut size = 2 + seps * 3;
        if tl != t0 {
            size += 1 + tl - t0;
        }

        let mut buf: Vec<u8> = vec![0; size];
        let mut n = buf.len();
        if n > 2 {
            n = 2;
        }

        for i in 0..n {
            buf[i] = '.' as u8;
        }

        let mut n = 2;
        for i in 0..seps {
            buf[i] = PathSeparator as u8;
            buf[n + 1] = '.' as u8;
            buf[n + 2] = '.' as u8;
            n += 3;
        }

        if t0 != tl {
            buf[n] = PathSeparator as u8;

            let to = &mut buf[n + 1..];
            let from = &targ[t0..];
            let mut min = to.len();
            if min > from.len() {
                min = from.len();
            }

            for i in 0..min {
                to[i] = from[i]
            }
        }

        return Ok(String::from_utf8(buf.to_vec()).unwrap());
    }

    return Ok(String::from_utf8(targ[t0..].to_vec()).unwrap());
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_Clean() {
        assert_eq!(Clean(&"/abc/../abc".to_string()), "/abc");
        assert_eq!(Clean(&"/abc/./.././abc".to_string()), "/abc");
        assert_eq!(Clean(&"/abc/../abc/.".to_string()), "/abc");
        assert_eq!(Clean(&"../abc/../abc/.".to_string()), "../abc");
        assert_eq!(Clean(&"../../abc/../abc/.".to_string()), "../../abc");
        assert_eq!(Clean(&"/abc/../../abc".to_string()), "/abc");
        assert_eq!(Clean(&"/abc/../xxxy/../../abc".to_string()), "/abc");
    }

    #[test]
    fn test_Split() {
        let path = "/abc/../abc".to_string();
        let (dir, file) = Split(&path);
        assert_eq!(dir.to_string(), "/abc/../");
        assert_eq!(file.to_string(), "abc");
    }

    #[test]
    fn test_Join() {
        assert_eq!(Join(&"/abc/..".to_string(), &"abc".to_string()), "/abc");
        assert_eq!(Join(&"/abc/..".to_string(), &"abc/".to_string()), "/abc");
    }

    #[test]
    fn test_Ext() {
        assert_eq!(Ext(&"/abc/../file.exe".to_string()), ".exe");
        assert_eq!(Ext(&"/abc/../file".to_string()), "");
    }

    #[test]
    fn test_Base() {
        assert_eq!(Base(&"/abc/../file.exe".to_string()), "file.exe");
        assert_eq!(Base(&"/abc/../file.exe/".to_string()), "file.exe");
        assert_eq!(Base(&"/abc/../file.exe///".to_string()), "file.exe");
        assert_eq!(Base(&"".to_string()), ".");
        assert_eq!(Base(&"asdf".to_string()), "asdf");
    }

    #[test]
    fn test_IsAbs() {
        assert_eq!(IsAbs(&"/abc/../file.exe".to_string()), true);
        assert_eq!(IsAbs(&"abc/../file.exe".to_string()), false);
    }

    #[test]
    fn test_Dir() {
        assert_eq!(Dir(&"/abc/xx/file.exe".to_string()), "/abc/xx");
        assert_eq!(Dir(&"/abc/../file.exe/".to_string()), "/file.exe");
        assert_eq!(Dir(&"/abc/../file.exe///".to_string()), "/file.exe");
        assert_eq!(Dir(&"".to_string()), ".");
        assert_eq!(Dir(&"asdf".to_string()), ".");
    }

    #[test]
    fn test_Rel() {
        assert_eq!(Rel("/abc", "/abc/def").unwrap(), "def");
        assert_eq!(Rel("/abc/", "/abc/def").unwrap(), "def");
        assert_eq!(Rel("/abc/../", "/abc/def").unwrap(), "abc/def");
        assert_eq!(Rel("/abc/../", "/abc/def/").unwrap(), "abc/def");
        assert_eq!(Rel("/abc/../", "/").unwrap(), ".");
    }
}

pub struct Path<'a> {
    pub arr: Vec<&'a str>,
    pub rawArr: Vec<&'a str>,
}

impl<'a> Path<'a> {
    pub fn New(path: &'a String) -> Self {
        let rawArr: Vec<&'a str> = path.split("/").collect();
        let mut arr = Vec::new();

        for name in &rawArr {
            if *name == "." || *name == "" {
                continue;
            }

            if *name == ".." {
                arr.pop();
            }

            arr.push(*name)
        }

        return Path { arr, rawArr };
    }

    pub fn IsPath(&self) -> bool {
        return self.rawArr.len() > 0 && self.rawArr[self.rawArr.len() - 1] == "";
    }

    pub fn IsAbs(&self) -> bool {
        return self.rawArr.len() > 0 && self.rawArr[0] == "";
    }
}

pub fn ParseBool(str: &str) -> Result<bool> {
    match str {
        "1" | "t" | "T" | "true" | "TRUE" | "True" => Ok(true),
        "0" | "f" | "F" | "false" | "FALSE" | "False" => Ok(false),
        _ => Err(Error::Common("parse error".to_string())),
    }
}
