use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use core::marker::PhantomData;
use core::ops::Deref;

pub struct ObjectRef <T> {
    addr: AtomicU64,
    obj: PhantomData<T>,
}

impl <T> Default for ObjectRef <T> {
    fn default() -> Self {
        return Self::New()
    }
}

impl <T> Deref for ObjectRef <T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*(self.addr.load(Ordering::Relaxed) as * const T)
        }
    }
}

impl <T> ObjectRef <T> {
    pub const fn New() -> Self {
        return Self {
            addr: AtomicU64::new(0),
            obj: PhantomData,
        }
    }

    pub fn Ptr(&self) -> &T {
        unsafe {
            &*(self.addr.load(Ordering::Relaxed) as * const T)
        }
    }

    pub fn SetValue(&self, addr: u64) {
        self.addr.store(addr, Ordering::SeqCst);
    }

    pub fn Value(&self) -> u64 {
        return self.addr.load(Ordering::Relaxed)
    }
}