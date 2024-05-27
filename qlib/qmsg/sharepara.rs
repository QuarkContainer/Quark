use super::super::linux_def::*;

#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub struct SharePara {
    pub para1: u64,
    pub para2: u64,
    pub para3: u64,
    pub para4: u64,
}

pub const SHAREPARA_COUNT: u64 = MemoryDef::PAGE_SIZE/core::mem::size_of::<SharePara>() as u64;

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct ShareParaPage {
    pub SharePara: [SharePara ;SHAREPARA_COUNT as usize],
}

impl Default for ShareParaPage{
    fn default() -> Self{
        return ShareParaPage{
            SharePara: [SharePara::default() ;SHAREPARA_COUNT as usize],
        }
    }
}