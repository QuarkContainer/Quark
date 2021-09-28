use core::mem::MaybeUninit;
use core::cell::UnsafeCell;
use core::ops::Deref;
//use core::format;

pub struct Singleton<T> {
    data: UnsafeCell<MaybeUninit<T>>,
}

impl <T> Deref for Singleton<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            self.force_get()
        }
    }
}

impl<T> Default for Singleton<T> {
    fn default() -> Self { Self::New() }
}

/*
impl<T: fmt::Debug> fmt::Debug for Singleton<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            Some(s) => write!(f, "Singleton {{ data: ")
                .and_then(|()| s.fmt(f))
                .and_then(|()| write!(f, "}}")),
            None => write!(f, "Singleton {{ <uninitialized> }}")
        }
    }
}*/

// Same unsafe impls as `std::sync::RwLock`, because this also allows for
// concurrent reads.
unsafe impl<T: Send + Sync> Sync for Singleton<T> {}
unsafe impl<T: Send> Send for Singleton<T> {}

impl <T> Singleton<T> {
    pub const fn New() -> Self {
        return Self {
            data: UnsafeCell::new(MaybeUninit::uninit())
        }
    }

    pub unsafe fn Init(&self, data: T) {
        let uninit = &mut *self.data.get();
        uninit.write(data);
    }

    /// Get a reference to the initialized instance. Must only be called once COMPLETE.
    unsafe fn force_get(&self) -> &T {
        // SAFETY:
        // * `UnsafeCell`/inner deref: data never changes again
        // * `MaybeUninit`/outer deref: data was initialized
        &*(*self.data.get()).as_ptr()
    }

    /// Get a reference to the initialized instance. Must only be called once COMPLETE.
    unsafe fn force_get_mut(&mut self) -> &mut T {
        // SAFETY:
        // * `UnsafeCell`/inner deref: data never changes again
        // * `MaybeUninit`/outer deref: data was initialized
        &mut *(*self.data.get()).as_mut_ptr()
    }
}