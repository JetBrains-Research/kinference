package org.jetbrains.research.kotlin.inference.extensions.primitives

operator fun FloatArray.unaryMinus(): FloatArray {
    val array = FloatArray(this.size)

    for (i in this.indices) array[i] = -this[i]
    return array
}


operator fun IntArray.unaryMinus(): IntArray {
    val array = IntArray(this.size)

    for (i in this.indices) array[i] = -this[i]
    return array
}

operator fun LongArray.unaryMinus(): LongArray {
    val array = LongArray(this.size)

    for (i in this.indices) array[i] = -this[i]
    return array
}

operator fun DoubleArray.unaryMinus(): DoubleArray {
    val array = DoubleArray(this.size)

    for (i in this.indices) array[i] = -this[i]
    return array
}

operator fun ShortArray.unaryMinus(): ShortArray {
    val array = ShortArray(this.size)

    for (i in this.indices) array[i] = (-this[i]).toShort()
    return array
}
