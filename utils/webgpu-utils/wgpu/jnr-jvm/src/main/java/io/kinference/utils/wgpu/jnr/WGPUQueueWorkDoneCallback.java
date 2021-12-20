package io.kinference.utils.wgpu.jnr;

import jnr.ffi.Pointer;
import jnr.ffi.annotations.Delegate;

public interface WGPUQueueWorkDoneCallback {
    @Delegate
    void callback(WGPUQueueWorkDoneStatus status, Pointer userdata);
}
