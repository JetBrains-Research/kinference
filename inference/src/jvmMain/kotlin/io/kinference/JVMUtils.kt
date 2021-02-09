package io.kinference

import okio.Buffer
import java.lang.Double.*
import java.lang.Float.*

actual fun Buffer.writeDouble(value: Double): Buffer = writeLong(doubleToLongBits(value))
actual fun Buffer.writeDoubleLe(value: Double): Buffer = writeLongLe(doubleToLongBits(value))

actual fun Buffer.readDouble(): Double = longBitsToDouble(readLong())
actual fun Buffer.readDoubleLe(): Double = longBitsToDouble(readLongLe())

actual fun Buffer.writeFloat(value: Float): Buffer = writeInt(floatToIntBits(value))
actual fun Buffer.writeFloatLe(value: Float): Buffer = writeIntLe(floatToIntBits(value))

actual fun Buffer.readFloat(): Float = intBitsToFloat(readInt())
actual fun Buffer.readFloatLe(): Float = intBitsToFloat(readIntLe())


