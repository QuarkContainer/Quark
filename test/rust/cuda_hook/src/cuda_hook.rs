use std::os::raw::{c_int, c_void};
use std::ffi::CString;
use cuda_runtime_sys::{dim3, cudaStream_t, cudaError_t};

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> cudaError_t {
    println!("Hijacked cudaSetDevice({})", device);

    let lib = CString::new("libcudart.so").unwrap();
    let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
    let func_name = CString::new("cudaSetDevice").unwrap();
    let orig_func: extern "C" fn(c_int) -> cudaError_t = unsafe {
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

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
        func: *const c_void, 
        grid_dim: dim3, 
        block_dim: dim3, 
        args: *mut *mut c_void, 
        shared_mem: usize, 
        stream: cudaStream_t
    ) -> cudaError_t {
    println!("Hijacked cudaLaunchKernel(grid_dim:({},{},{}), shared_mem:({},{},{}), shared_mem:{})", grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z, shared_mem);

    let lib = CString::new("libcudart.so").unwrap();
    let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
    let func_name = CString::new("cudaLaunchKernel").unwrap();
    let orig_func: extern "C" fn(*const c_void, dim3, dim3, *mut *mut c_void, usize, cudaStream_t) -> cudaError_t = unsafe {
        std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
    };
    orig_func(func, grid_dim, block_dim, args, shared_mem, stream)
}