package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.allocateDirect
import jnr.ffi.Pointer

fun ByteArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this, 0, size)
}

fun ShortArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this, 0, size)
}

fun IntArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this, 0, size)
}

fun LongArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this, 0, size)
}

fun FloatArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this, 0, size)
}

fun DoubleArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this, 0, size)
}

fun UByteArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this.asByteArray(), 0, size)
}

fun UShortArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this.asShortArray(), 0, size)
}

fun UIntArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this.asIntArray(), 0, size)
}

fun ULongArray.createPointerTo(): Pointer = allocateDirect(sizeBytes).also { pointer ->
    pointer.put(0, this.asLongArray(), 0, size)
}
