package io.kinference.tfjs.utils

import io.kinference.tfjs.custom_externals.backend_cpu.MathBackendCPU
import io.kinference.tfjs.custom_externals.backend_webgl.MathBackendWebGL
import io.kinference.tfjs.custom_externals.core.*
import io.kinference.ndarray.runBlocking
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.await


internal fun setWebGLBackend() {
    if (getBackend() != "webgl") {
        removeBackend("webgl")
        registerBackend("webgl", { MathBackendWebGL(null) })
        runBlocking(Dispatchers.Default) { setBackend("webgl").await() }
    }
}

internal fun setCPUBackend() {
    if (getBackend() != "cpu") {
        removeBackend("cpu")
        registerBackend("cpu", { MathBackendCPU() })
        runBlocking(Dispatchers.Default) { setBackend("cpu").await() }
    }
}

internal fun setDefaultBackend() {
    setWebGLBackend()
}
