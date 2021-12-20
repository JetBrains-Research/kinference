package io.kinference.utils.webgpu

import org.khronos.webgl.*

actual class BufferData(val buffer: ArrayBuffer) {
    actual val size = buffer.byteLength

    actual constructor(array: IntArray) : this(ArrayBuffer(array.sizeBytes)) { set(array) }
    actual constructor(array: FloatArray) : this(ArrayBuffer(array.sizeBytes)) { set(array) }

    actual fun set(data: BufferData) {
        Int8Array(buffer).set(Int8Array(data.buffer))
    }

    actual fun set(array: IntArray) {
        Int32Array(buffer).set(array.unsafeCast<Int32Array>())
    }

    actual fun set(array: FloatArray) {
        Float32Array(buffer).set(array.unsafeCast<Float32Array>())
    }

    actual fun toIntArray(): IntArray = Int32Array(buffer).unsafeCast<IntArray>().copyOf()

    actual fun toUIntArray(): UIntArray = Uint32Array(buffer).unsafeCast<UIntArray>().copyOf()

    actual fun toFloatArray(): FloatArray = Float32Array(buffer).unsafeCast<FloatArray>().copyOf()
}
