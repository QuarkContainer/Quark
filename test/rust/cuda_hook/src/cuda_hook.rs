use std::os::raw::{c_int, c_void};
use std::ffi::CString;

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> c_int {
    println!("Hijacked cudaSetDevice({})", device);

    let lib = CString::new("libcudart.so").unwrap();
    let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
    let func_name = CString::new("cudaSetDevice").unwrap();
    let orig_func: extern "C" fn(c_int) -> c_int = unsafe {
        std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
    };
    orig_func(device)
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> c_void {
    println!("Hijacked cudaDeviceSynchronize()");

    let lib = CString::new("libcudart.so").unwrap();
    let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
    let func_name = CString::new("cudaDeviceSynchronize").unwrap();
    let orig_func: extern "C" fn() -> c_void = unsafe {
        std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
    };
    orig_func()
}