@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.arrays

import io.kinference.primitives.types.PrimitiveArray


import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import io.kinference.utils.inlines.InlineInt
import kotlin.jvm.JvmName

@GenerateNameFromPrimitives
@MakePublic
internal open class MutablePrimitiveLNDArray(array: PrimitiveArray, strides: Strides = Strides.EMPTY) : PrimitiveLNDArray(array, strides) {
    fun set(index: IntArray, value: Any) {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        array[linearIndex] = value as PrimitiveType
    }

    fun setLinear(index: Int, value: Any) {
        array[index] = value as PrimitiveType
    }

    fun fill(value: Any, from: Int = 0, to: Int = linearSize) {
        array.fill(value as PrimitiveType, from, to)
    }

    operator fun plusAssign(other: PrimitiveLNDArray) {
        for (i in 0 until linearSize) {
            array[i] = (array[i] + other.array[i]).toPrimitive()
        }
    }

    operator fun minusAssign(other: PrimitiveLNDArray) {
        for (i in 0 until linearSize) {
            array[i] = (array[i] - other.array[i]).toPrimitive()
        }
    }

    operator fun timesAssign(other: PrimitiveLNDArray) {
        for (i in 0 until linearSize) {
            array[i] = (array[i] * other.array[i]).toPrimitive()
        }
    }

    operator fun divAssign(other: PrimitiveLNDArray) {
        for (i in 0 until linearSize) {
            array[i] = (array[i] / other.array[i]).toPrimitive()
        }
    }

    fun clean() {
        array.fill(0.toPrimitive())
    }
}
