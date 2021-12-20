package io.kinference.utils.wgpu.jnr;

import jnr.ffi.Pointer;
import jnr.ffi.annotations.Delegate;

public interface WGPURequestAdapterCallback {
    @Delegate
    void callback(WGPURequestAdapterStatus status, Pointer adapter, Pointer message, Pointer userdata);
}
