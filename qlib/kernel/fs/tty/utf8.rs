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

const XX: u8 = 0xF1; // invalid: size 1
const AS: u8 = 0xF0; // ASCII: size 1
const S1: u8 = 0x02; // accept 0, size 2
const S2: u8 = 0x13; // accept 1, Size 3
const S3: u8 = 0x03; // accept 0, size 3
const S4: u8 = 0x23; // accept 2, size 3
const S5: u8 = 0x34; // accept 3, size 4
const S6: u8 = 0x04; // accept 0, size 4
const S7: u8 = 0x44; // accept 4, size 4

// The default lowest and highest continuation byte.
const LOCB: u8 = 0x80; // 1000 0000
const HICB: u8 = 0xBF; // 1011 1111

const FIRST: [u8; 256] = [
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x00-0x0F
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x10-0x1F
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x20-0x2F
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x30-0x3F
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x40-0x4F
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x50-0x5F
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x60-0x6F
    AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, AS, // 0x70-0x7F
    //   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, // 0x80-0x8F
    XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, // 0x90-0x9F
    XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, // 0xA0-0xAF
    XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, // 0xB0-0xBF
    XX, XX, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, // 0xC0-0xCF
    S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, S1, // 0xD0-0xDF
    S2, S3, S3, S3, S3, S3, S3, S3, S3, S3, S3, S3, S3, S4, S3, S3, // 0xE0-0xEF
    S5, S6, S6, S6, S7, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, XX, // 0xF0-0xFF
];

#[derive(Copy, Clone)]
struct AcceptRange {
    pub lo: u8,
    pub hi: u8,
}

const ACCEPT_RANGES: [AcceptRange; 5] = [
    AcceptRange { lo: LOCB, hi: HICB },
    AcceptRange { lo: 0xA0, hi: HICB },
    AcceptRange { lo: LOCB, hi: 0x9F },
    AcceptRange { lo: 0x90, hi: HICB },
    AcceptRange { lo: LOCB, hi: 0x8F },
];

pub fn DecodeUtf8(p: &[u8]) -> usize {
    let n = p.len();

    if n < 1 {
        return 0
    }

    let p0 = p[0] as usize;
    let x = FIRST[p0];
    if x > AS {
        return 1;
    }

    let sz = x & 7;
    let accept = ACCEPT_RANGES[(x >> 7) as usize];
    if n < sz as usize {
        return 1;
    }

    let b1 = p[1];
    if b1 < accept.lo || accept.hi < b1 {
        return 1;
    }

    if sz == 2 {
        return 2;
    }

    let b2 = p[2];
    if b2 < LOCB || HICB < b2 {
        return 1;
    }

    if sz == 3 {
        return 3;
    }

    let b3 = p[3];
    if b3 < LOCB || HICB < b3 {
        return 1
    }

    return 4;
}