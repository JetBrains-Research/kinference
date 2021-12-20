package io.kinference.utils.wgpu.jnr;

import jnr.ffi.Pointer;
import jnr.ffi.annotations.Delegate;

public interface WGPURequestDeviceCallback {
    @Delegate
    void callback(WGPURequestDeviceStatus status, Pointer device, Pointer message, Pointer userdata);
}
