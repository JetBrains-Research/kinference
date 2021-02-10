package io.kinference.utils

import okio.Buffer
import org.khronos.webgl.*

actual fun Buffer.writeDouble(value: Double): Buffer {
    val arrayBuffer = ArrayBuffer(8)
    Float64Array(arrayBuffer).apply { set(0, value) }
    val byteBuffer = Uint8Array(arrayBuffer)
    for (i in 0 until 8) {
        writeByte(byteBuffer[i].toInt())
    }
    return this
}

actual fun Buffer.writeDoubleLe(value: Double): Buffer {
    val arrayBuffer = ArrayBuffer(8)
    Float64Array(arrayBuffer).apply { set(0, value) }
    val byteBuffer = Uint8Array(arrayBuffer)
    for (i in 7 downTo 0) {
        writeByte(byteBuffer[i].toInt())
    }
    return this
}

actual fun Buffer.readDouble(): Double {
    val arrayBuffer = ArrayBuffer(8)
    val byteBuffer = Uint8Array(arrayBuffer)

    for (i in 0 until 8) {
        byteBuffer[i] = readByte()
    }

    return Float64Array(arrayBuffer)[0]
}
actual fun Buffer.readDoubleLe(): Double {
    val arrayBuffer = ArrayBuffer(8)
    val byteBuffer = Uint8Array(arrayBuffer)

    for (i in 7 downTo 0) {
        byteBuffer[i] = readByte()
    }

    return Float64Array(arrayBuffer)[0]
}

actual fun Buffer.writeFloat(value: Float): Buffer {
    val arrayBuffer = ArrayBuffer(4)
    Float32Array(arrayBuffer).apply { set(0, value) }
    val intBuffer = Uint32Array(arrayBuffer)
    return writeInt(intBuffer[0])
}

actual fun Buffer.writeFloatLe(value: Float): Buffer {
    val arrayBuffer = ArrayBuffer(4)
    Float32Array(arrayBuffer).apply { set(0, value) }
    val intBuffer = Uint32Array(arrayBuffer)
    return writeIntLe(intBuffer[0])
}

actual fun Buffer.readFloat(): Float {
    val arrayBuffer = ArrayBuffer(4)
    Uint32Array(arrayBuffer)[0] = readInt()

    return Float32Array(arrayBuffer)[0]
}

actual fun Buffer.readFloatLe(): Float {
    val arrayBuffer = ArrayBuffer(4)
    Uint32Array(arrayBuffer)[0] = readIntLe()

    return Float32Array(arrayBuffer)[0]
}
