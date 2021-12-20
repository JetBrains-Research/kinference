package io.kinference.utils.wgpu.jnr;

import jnr.ffi.Pointer;
import jnr.ffi.annotations.Delegate;

public interface WGPUBufferMapCallback {
    @Delegate
    void callback(WGPUBufferMapAsyncStatus status, Pointer userdata);
}
