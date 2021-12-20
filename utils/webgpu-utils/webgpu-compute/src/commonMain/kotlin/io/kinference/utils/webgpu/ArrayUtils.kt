package io.kinference.utils.webgpu

val ByteArray.sizeBytes
    get() = size * Byte.SIZE_BYTES

val ShortArray.sizeBytes
    get() = size * Short.SIZE_BYTES

val IntArray.sizeBytes
    get() = size * Int.SIZE_BYTES

val LongArray.sizeBytes
    get() = size * Long.SIZE_BYTES

val FloatArray.sizeBytes
    get() = size * Float.SIZE_BYTES

val DoubleArray.sizeBytes
    get() = size * Double.SIZE_BYTES

val UByteArray.sizeBytes
    get() = size * UByte.SIZE_BYTES

val UShortArray.sizeBytes
    get() = size * UShort.SIZE_BYTES

val UIntArray.sizeBytes
    get() = size * UInt.SIZE_BYTES

val ULongArray.sizeBytes
    get() = size * ULong.SIZE_BYTES
