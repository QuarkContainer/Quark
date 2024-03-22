use std::os::raw::*;
use std::ffi::CString;
use cuda_runtime_sys::{dim3, cudaStream_t, cudaError_t, cudaMemcpyKind};
use libc::dlsym;
use std::mem::transmute;
use std::ptr;

pub static mut DLOPEN_ORIG: Option<unsafe extern "C" fn(*const libc::c_char, libc::c_int) -> *mut libc::c_void> = None;
pub static mut DLCLOSE_ORIG: Option<unsafe extern "C" fn(*mut libc::c_void) -> libc::c_int> = None;
pub static mut DL_HANDLE: *mut libc::c_void = ptr::null_mut();

#[no_mangle]
pub extern "C" fn dlopen(filename: *const libc::c_char, flag: libc::c_int) -> *mut c_void {
     let mut ret: *mut c_void  = std::ptr::null_mut();

    let c_str = unsafe {std::ffi::CStr::from_ptr(filename) };
    let filename_string = c_str.to_string_lossy().to_string();
    println!("intercepted dlopen({} {})", filename_string, flag);
    
    // let ret = unsafe{::libc::dlopen(filename, flag) };
    // println!("ret {:x?}",ret);
    // return ret;
    
    // if filename.is_null() {
    //     return unsafe { DLOPEN_ORIG.unwrap()(filename, flag) };
       
    // }
    
    if unsafe { DLOPEN_ORIG.is_none() } {
        let symbol = CString::new("dlopen").unwrap();
        // let orig = unsafe { dlsym(libc::RTLD_NEXT, symbol.as_ptr()) };

        // let a : fn(*const libc::c_char, libc::c_int) -> *mut libc::c_void = unsafe{  std::mem::transmute(
        //     ::libc::dlsym(::libc::RTLD_NEXT, std::mem::transmute(symbol.as_ptr())) ) };
        // let a : fn(*const libc::c_char, libc::c_int) -> *mut libc::c_void = unsafe{  std::mem::transmute(
        //       ::libc::dlopen(filename, flag)) };
        // ret = a(filename, flag) ;

        // println!("DLopen_Orig is {:x?}", ret);
        // return ret;
      
         unsafe {DLOPEN_ORIG =  std::mem::transmute(
              ::libc::dlsym(::libc::RTLD_NEXT, std::mem::transmute(b"dlopen\x00".as_ptr()) )) };
        
        // unsafe {DLOPEN_ORIG = Some(std::mem::transmute(::libc::dlopen(filename, flag))) };

        
        // if orig.is_null() {
        //     println!("[dlopen] dlsym failed");
        // } else {
        //     // unsafe{ DLOPEN_ORIG =  Some(std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_char, c_int) -> *mut c_void>(orig)) };
        //     println!("11111111 dlsym successfulely");
        //     unsafe {
        //         DLOPEN_ORIG = Some(transmute(orig));
        //     }
        // }
    }
    if unsafe { DLOPEN_ORIG.is_none() } {
        println!("DLopen_Orig is still none");
        
    }

    // let real_rand = unsafe { DLOPEN_ORIG.unwrap_unchecked() };
    // ret = unsafe { real_rand(filename, flag) };
    // println!("DLopen_Orig is {:x?}", ret);
     // ret = unsafe { DLOPEN_ORIG.unwrap()(filename, flag) };
    // return ret;

    let replace_libs = [
        "libcuda.so.1",
        "libcuda.so",
        "libnvidia-ml.so.1",
        "libcudnn_cnn_infer.so.8",
    ];

    if !filename.is_null() {
        for libs in &replace_libs{
            if filename_string == *libs {
                println!("replacing dlopen call to {} with libcudaproxy.so", libs);
                let cudaProxy = CString::new("libcuda_hook.so").unwrap();
                unsafe {
                    DL_HANDLE = DLOPEN_ORIG.unwrap()(
                        cudaProxy.as_ptr(),
                        flag,
                    );
                }
                if unsafe { DL_HANDLE.is_null() } {
                    println!("failed to replaced dlooen call to libcudaproxy.so");
                }
                return unsafe { DL_HANDLE };
            }
        }
    }

    ret = unsafe { DLOPEN_ORIG.unwrap()(filename, flag) };
    println!("22222222");

    // println!("value of ret {:#?}", ret);

    // if ret.is_null() {
    //     println!("2");
    //     let err = unsafe { libc::dlerror() };
    //     let errMesg = unsafe { CString::from_raw(err) };
    //     println!(
    //         "dlopen {} failed: {}",
    //         filename_string,
    //         errMesg.to_str().unwrap_or("unknown error")
    //     );
    // }
    // println!("3");
     return ret;
}


#[no_mangle]
pub extern "C" fn dlclose(handle: *mut c_void) -> c_int{
    if handle.is_null() {
        println!("[dlclose] handle NULL");
        return -1;
    }else if unsafe{ DLCLOSE_ORIG.is_none() } {
        let symbol= CString::new("dlclose").unwrap();
        let orig = unsafe { dlsym(libc::RTLD_NEXT, symbol.as_ptr()) };
        if orig.is_null(){
            println!("[dlclose] dlsym failed");
        }else {
            // unsafe{ DLCLOSE_ORIG =  Some(std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_void) -> *mut c_void>(orig)) };
            unsafe {
                DLCLOSE_ORIG = Some(transmute(orig));
            }
        }
    }
    if unsafe{ DL_HANDLE == handle } {
        println!("[dlclose] ignore close");
        return 0;
    } else{
        return unsafe{ DLCLOSE_ORIG.unwrap()(handle) };
    }
}

// #[no_mangle]
// pub extern "C" fn cudaSetDevice(device: c_int) -> cudaError_t {
//     println!("Hijacked cudaSetDevice({})", device);

//     let lib = CString::new("libcudart.so").unwrap();
//     let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
//     let func_name = CString::new("cudaSetDevice").unwrap();
//     let orig_func: extern "C" fn(c_int) -> cudaError_t = unsafe {
//         std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
//     };
//     orig_func(device)
// }

// #[no_mangle]
// pub extern "C" fn cudaDeviceSynchronize() -> c_void {
//     println!("Hijacked cudaDeviceSynchronize()");

//     let lib = CString::new("libcudart.so").unwrap();
//     let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
//     let func_name = CString::new("cudaDeviceSynchronize").unwrap();
//     let orig_func: extern "C" fn() -> c_void = unsafe {
//         std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
//     };
//     orig_func()
//}

// #[no_mangle]
// pub extern "C" fn __cudaRegisterFatBinary(fatCubin: u64) -> *mut u64 {
//     println!("Hijacked __cudaRegisterFatBinary(fatCubin:{:x})", fatCubin);
//      let lib = CString::new("libcudart.so").unwrap();
//     let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
//     let func_name = CString::new("__cudaRegisterFatBinary").unwrap();
//     let orig_func: extern "C" fn(*const c_void, dim3, dim3, *mut *mut c_void, usize, cudaStream_t) -> cudaError_t = unsafe {
//         std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
//     };
//     orig_func(func, grid_dim, block_dim, args, shared_mem, stream)
//     return 0;
//     // let tempVaue = 0;
//     // let result = &tempVaue as *const _ as u64;
//     // return result as *mut u64;
  
// }

// #[no_mangle]
// pub extern "C" fn __cudaRegisterVar(
//     fatCubinHandle: u64,
//     hostVar:u64,
//     deviceAddress: u64,
//     deviceName: u64,
//     ext: c_int,
//     size: u64,
//     constant: c_int,
//     global: c_int
// ){
//     println!("Hijacked __cudaRegisterVar(fatCubinHandle: {:x},hostVar: {:x}, deviceAddress: {:x}, deviceName:{:x} )",fatCubinHandle,hostVar, deviceAddress,deviceName);
// }


// #[no_mangle]
// pub extern "C" fn __cudaRegisterFunction(
//     fatCubinHandle:u64,
//     hostFun: u64,
//     deviceFun:u64, 
//     deviceName: u64,
//     thread_limit: c_int,
//     tid: u64,
//     bid:u64,
//     bDim: u64,
//     gDim: u64,
//     wSize: *mut c_int
// ){
//     println!("Hijacked __cudaRegisterFunction(fatCubinHandle: {},hostFun: {}, deviceFun: {}, deviceName:{} )",fatCubinHandle,hostFun, deviceFun,deviceName);
// }

// #[no_mangle]
// pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle:u64) {
//     println!("Hijacked __cudaUnregisterFatBinary(fatCubinHandle = {:x})", fatCubinHandle);   
// }


// #[no_mangle]
// pub extern "C" fn __cudaRegisterFatBinaryEnd(fatCubinHandle : u64){
//     println!("Hijacked __cudaUnregisterFatBinaryEnd(fatCubinHandle = {:x})", fatCubinHandle);   
// }


// #[no_mangle]
// pub extern "C" fn cudaLaunchKernel(
//         func: *const c_void, 
//         grid_dim: dim3, 
//         block_dim: dim3, 
//         args: *mut *mut c_void, 
//         shared_mem: usize, 
//         stream: cudaStream_t
//     ) -> i64 {
//     println!("Hijacked cudaLaunchKernel(grid_dim:({},{},{}), shared_mem:({},{},{}), shared_mem:{})", grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z, shared_mem);

//     // let lib = CString::new("libcudart.so").unwrap();
//     // let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
//     // let func_name = CString::new("cudaLaunchKernel").unwrap();
//     // let orig_func: extern "C" fn(*const c_void, dim3, dim3, *mut *mut c_void, usize, cudaStream_t) -> cudaError_t = unsafe {
//     //     std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
//     // };
//     // orig_func(func, grid_dim, block_dim, args, shared_mem, stream)
//     return 0;
// }

// #[no_mangle]
// pub extern "C" fn cuDevicePrimaryCtxGetState(dev: c_int, flags: *mut c_uint,  active: *mut c_int) -> i64{
//     println!("Hijacked cuDevicePrimaryCtxGetState(dev: {:x}",dev);
//     // let lib = CString::new("libcudart.so").unwrap();
//     // let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
//     // let func_name = CString::new("cudaLaunchKernel").unwrap();
//     // let orig_func: extern "C" fn(*const c_void, dim3, dim3, *mut *mut c_void, usize, cudaStream_t) -> cudaError_t = unsafe {
//     //     std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
//     // };
//     // orig_func(func, grid_dim, block_dim, args, shared_mem, stream)

//     return 0;
// }


// #[no_mangle]
// pub extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> i64{
//     println!("Hijacked cudaGetDeviceCount");
//     return 0;
// }

// #[no_mangle]
// pub extern "C" fn cudaGetDevice(count:*mut c_int) -> i64{
//     println!("Hijacked cudaGetDevice");
    
//     unsafe{* (count as *mut _) = 0};
//     return 0;
// }
// #[no_mangle]
// pub extern "C" fn cuInit( Flags: c_uint) -> i64{
//     println!("Hijacked cuInit");
//     return 0;
// }

// #[no_mangle]
// pub extern "C" fn cudaDeviceGetStreamPriorityRange(leastPriority:*mut c_int,greatestPriority: *mut c_int) -> i64{
//     println!("Hijacked cudaDeviceGetStreamPriorityRange");
//     return 0;
// }




// #[no_mangle]
// pub extern "C" fn cudaMalloc(
//         dev_ptr: *mut *mut c_void, 
//         size: usize
//     ) -> cudaError_t {
//     println!("Hijacked cudaMalloc(size:{})", size);

//     let lib = CString::new("libcudart.so").unwrap();
//     let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
//     let func_name = CString::new("cudaMalloc").unwrap();
//     let orig_func: extern "C" fn(*mut *mut c_void, usize) -> cudaError_t = unsafe {
//         std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
//     };
//     orig_func(dev_ptr, size)
// }

// #[no_mangle]
// pub extern "C" fn cudaMemcpy(
//         dst: *mut c_void, 
//         src: *const c_void, 
//         count: usize, 
//         kind: cudaMemcpyKind
//     ) -> cudaError_t {
//     println!("Hijacked cudaMemcpy(count:{})", count);

//     let lib = CString::new("libcudart.so").unwrap();
//     let handle = unsafe { libc::dlopen(lib.as_ptr(), libc::RTLD_LAZY) };    
//     let func_name = CString::new("cudaMemcpy").unwrap();
//     let orig_func: extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> cudaError_t = unsafe {
//         std::mem::transmute(libc::dlsym(handle, func_name.as_ptr()))
//     };
//     orig_func(dst, src, count, kind)
// }