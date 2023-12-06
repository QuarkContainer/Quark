use std::os::raw::{c_int, c_void};
use std::ffi::CString;
use cuda_runtime_sys::{dim3, cudaStream_t, cudaError_t, cudaMemcpyKind};

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

#[no_mangle]
pub extern "C" fn cudaMalloc(
        dev_ptr: *mut *mut c_void, 
        size: usize
    ) -> cudaError_t {
    println!("Hijacked cudaMalloc(size:{})", size);

    let lib = CString::new("libcudart.so").unwrap();
    let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
    let func_name = CString::new("cudaMalloc").unwrap();
    let orig_func: extern "C" fn(*mut *mut c_void, usize) -> cudaError_t = unsafe {
        std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
    };
    orig_func(dev_ptr, size)
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
        dst: *mut c_void, 
        src: *const c_void, 
        count: usize, 
        kind: cudaMemcpyKind
    ) -> cudaError_t {
    println!("Hijacked cudaMemcpy(count:{})", count);

    let lib = CString::new("libcudart.so").unwrap();
    let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
    let func_name = CString::new("cudaMemcpy").unwrap();
    let orig_func: extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> cudaError_t = unsafe {
        std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
    };
    orig_func(dst, src, count, kind)
}