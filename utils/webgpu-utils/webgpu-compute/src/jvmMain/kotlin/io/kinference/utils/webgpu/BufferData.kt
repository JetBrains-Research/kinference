package io.kinference.utils.webgpu

import jnr.ffi.Pointer

actual class BufferData(val pointer: Pointer, actual val size: Int) {
    actual constructor(array: IntArray) : this(array.createPointerTo(), array.sizeBytes)
    actual constructor(array: FloatArray) : this(array.createPointerTo(), array.sizeBytes)

    actual fun set(data: BufferData) {
        val bytes = ByteArray(size)
        data.pointer.get(0, bytes, 0, size)
        pointer.put(0, bytes, 0, size)
    }

    actual fun set(array: IntArray) {
        pointer.put(0, array, 0, array.size)
    }

    actual fun set(array: FloatArray) {
        pointer.put(0, array, 0, array.size)
    }

    actual fun toIntArray(): IntArray = IntArray(size / Int.SIZE_BYTES).also { array ->
        pointer.get(0, array, 0, array.size)
    }

    actual fun toUIntArray(): UIntArray = toIntArray().asUIntArray()

    actual fun toFloatArray(): FloatArray = FloatArray(size / Float.SIZE_BYTES).also { array ->
        pointer.get(0, array, 0, array.size)
    }
}
