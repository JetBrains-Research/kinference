package io.kinference.utils

import okio.Buffer



expect fun Buffer.writeDouble(value: Double): Buffer
expect fun Buffer.writeDoubleLe(value: Double): Buffer

expect fun Buffer.readDouble(): Double
expect fun Buffer.readDoubleLe(): Double

expect fun Buffer.writeFloat(value: Float): Buffer
expect fun Buffer.writeFloatLe(value: Float): Buffer

expect fun Buffer.readFloat(): Float
expect fun Buffer.readFloatLe(): Float
