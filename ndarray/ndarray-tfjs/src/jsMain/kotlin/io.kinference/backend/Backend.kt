package io.kinference.backend

import io.kinference.utils.runBlocking
import io.kinference.backend.cpu.MathBackendCPU
import io.kinference.backend.webgl.MathBackendWebGL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.await


fun setWebGLBackend() {
    if (getBackend() != "webgl") {
        removeBackend("webgl")
        registerBackend("webgl", { MathBackendWebGL(null) })
        runBlocking(Dispatchers.Default) { setBackend("webgl").await() }
    }
}

fun setCPUBackend() {
    if (getBackend() != "cpu") {
        removeBackend("cpu")
        registerBackend("cpu", { MathBackendCPU() })
        runBlocking(Dispatchers.Default) { setBackend("cpu").await() }
    }
}

fun setDefaultBackend() {
    setWebGLBackend()
}
