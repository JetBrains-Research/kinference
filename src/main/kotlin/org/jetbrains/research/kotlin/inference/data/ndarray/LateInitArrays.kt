package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.math.LateInitArray

class LateInitFloatArray(size: Int) : LateInitArray {
    private val array = FloatArray(size)
    private var index = 0

    fun putNext(value: Float) {
        array[index] = value
        index++
    }

    fun getArray(): FloatArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}

class LateInitDoubleArray(size: Int) : LateInitArray {
    private val array = DoubleArray(size)
    private var index = 0

    fun putNext(value: Double) {
        array[index] = value
        index++
    }

    fun getArray(): DoubleArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}

class LateInitIntArray(size: Int) : LateInitArray {
    private val array = IntArray(size)
    private var index = 0

    fun putNext(value: Int) {
        array[index] = value
        index++
    }

    fun getArray(): IntArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}

class LateInitLongArray(size: Int) : LateInitArray {
    private val array = LongArray(size)
    private var index = 0

    fun putNext(value: Long) {
        array[index] = value
        index++
    }

    fun getArray(): LongArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}

class LateInitShortArray(size: Int) : LateInitArray {
    private val array = ShortArray(size)
    private var index = 0

    fun putNext(value: Short) {
        array[index] = value
        index++
    }

    fun getArray(): ShortArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}

class LateInitBooleanArray(size: Int) : LateInitArray {
    private val array = BooleanArray(size)
    private var index = 0

    fun putNext(value: Boolean) {
        array[index] = value
        index++
    }

    fun getArray(): BooleanArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}
