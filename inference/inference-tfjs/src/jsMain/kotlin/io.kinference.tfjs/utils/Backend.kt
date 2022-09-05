package io.kinference.tfjs.utils

import io.kinference.utils.runBlocking
import io.kinference.tfjs.externals.backend.cpu.MathBackendCPU
import io.kinference.tfjs.externals.backend.webgl.MathBackendWebGL
import io.kinference.tfjs.externals.core.*
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
