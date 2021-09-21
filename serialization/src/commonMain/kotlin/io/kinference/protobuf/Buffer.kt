package io.kinference.protobuf

import okio.Buffer

fun Buffer.writeDouble(value: Double): Buffer = writeLong(value.toRawBits())
fun Buffer.writeDoubleLe(value: Double): Buffer = writeLongLe(value.toRawBits())

fun Buffer.readDouble(): Double = Double.fromBits(readLong())
fun Buffer.readDoubleLe(): Double = Double.fromBits(readLongLe())


fun Buffer.writeFloat(value: Float): Buffer = writeInt(value.toRawBits())
fun Buffer.writeFloatLe(value: Float): Buffer = writeIntLe(value.toRawBits())

fun Buffer.readFloat(): Float = Float.fromBits(readInt())
fun Buffer.readFloatLe(): Float = Float.fromBits(readIntLe())
