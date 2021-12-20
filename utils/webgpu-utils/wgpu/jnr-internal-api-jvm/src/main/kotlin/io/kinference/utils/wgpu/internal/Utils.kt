package io.kinference.utils.wgpu.internal

import jnr.ffi.Pointer
import jnr.ffi.Struct
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.US_ASCII

val nullptr: Pointer
    get() = Pointer.wrap(WgpuRuntime.runtime, 0x00L)

val Pointer.isNullptr
    get() = address() == 0x00L

fun String.createPointerTo(): Pointer {
    val buffer = ByteBuffer.allocateDirect(length + 1)
    buffer.put(toByteArray(US_ASCII))
    buffer.put(0x00)
    buffer.position(0)

    return Pointer.wrap(WgpuRuntime.runtime, buffer)
}

fun Pointer.getString(): String {
    val outputStream = ByteArrayOutputStream()

    for (i in 0..Int.MAX_VALUE) {
        val nextChar = getByte(i.toLong()).toInt()
        outputStream.write(nextChar)
        if (nextChar == 0x00) {
            break
        }
    }
    return outputStream.toString()
}

fun Struct.useDirectMemory() {
    useMemory(allocateDirect(Struct.size(this)))
}

fun Struct.getPointerTo(): Pointer = Struct.getMemory(this)

fun allocateDirect(size: Int): Pointer = WgpuRuntime.runtime.memoryManager.allocateDirect(size)
