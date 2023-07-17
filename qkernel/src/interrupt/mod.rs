
cfg_x86_64! {
    mod x86_64;
    pub use self::x86_64::*;
}

cfg_aarch64! {
    mod aarch64;
    pub use self::aarch64::*;
}