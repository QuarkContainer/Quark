pub mod sev_snp;

pub enum VmType{
    Normal,
    SevSnp,
    CCEmu,
}
impl Default for VmType {
    fn default() -> Self {
        VmType::Normal
    }
}

impl VmType{
    pub fn from_u64(value: u64) -> Option<VmType> {
        match value {
            0 => Some(VmType::Normal),
            1 => Some(VmType::SevSnp),
            2 => Some(VmType::CCEmu),
            _ => None,
        }
    }

    pub fn to_u64(&self) -> u64 {
        match self {
            VmType::Normal => 0,
            VmType::SevSnp => 1,
            VmType::CCEmu => 2,
        }
    }
}