@file:GeneratePrimitives(DataType.NUMBER)
@file:Suppress("DuplicatedCode", "unused")

package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import io.kinference.ndarray.stubs.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.extensions.dot.DotUtils
import io.kinference.ndarray.stubs.MAX_VALUE_FOR_MIN
import io.kinference.ndarray.stubs.MIN_VALUE_FOR_MAX
import io.kinference.utils.inlines.InlineInt
import kotlin.math.PI

@GenerateNameFromPrimitives
@MakePublic
internal open class PrimitiveLNDArray(array: PrimitiveArray, strides: Strides) {
    var array: PrimitiveArray = array
    final var strides: Strides = strides
        protected set

    val shape: IntArray
        get() = strides.shape

    val rank: Int
        get() = shape.size

    val linearSize: Int
        get() = strides.linearSize

    operator fun get(index: IntArray): PrimitiveType {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        return array[linearIndex]
    }

    fun getLinear(index: Int): PrimitiveType {
        return array[index]
    }

    val type = DataType.CurrentPrimitive

    suspend fun clone(): PrimitiveLNDArray {
        return PrimitiveLNDArray(array.copyOf(), Strides(shape))
    }

    suspend fun toMutable(): MutablePrimitiveLNDArray = MutablePrimitiveLNDArray(array.copyOf(), strides)

    suspend fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutablePrimitiveLNDArray {
        function as PrimitiveMap
        destination as MutablePrimitiveLNDArray
        val src = this.array
        val dest = destination.array
        for (index in 0 until linearSize) {
            dest[index] = function.apply(src[index])
        }
        return destination
    }

    suspend fun sum(): PrimitiveType {
        var sum = 0.toPrimitive()
        for (index in 0 until linearSize) {
            sum = (sum + array[index]).toPrimitive()
        }
        return sum
    }

    suspend fun max(): PrimitiveType {
        var max = PrimitiveType.MIN_VALUE_FOR_MAX
        for (index in 0 until linearSize) {
            if (array[index] > max) max = array[index]
        }
        return max
    }

    suspend fun min(): PrimitiveType {
        var min = PrimitiveType.MAX_VALUE_FOR_MIN
        for (index in 0 until linearSize) {
            if (array[index] < min) min = array[index]
        }
        return min
    }

    suspend fun dot(other: PrimitiveLNDArray, dest: MutablePrimitiveLNDArray): MutablePrimitiveLNDArray {
        require(shape.size in 1..2 && other.shape.size in 1..2)
        require(shape[1] == other.shape[0])

        val n = shape[0]
        val t = shape[1]
        val m = other.shape[1]
        val nRowFlop = t * m

        val destArray = dest.array
        val leftArray = this.array
        val rightArray = other.array

        val blockSize = 64
        val rMax = m - (m % blockSize)

        parallelizeByRows(nRowFlop, n,2* 131072) { nStart, nEnd, _ ->
            for (i in nStart until nEnd) {
                for(offset in 0 until rMax step blockSize) {
                    val destBlockOffset = i * m + offset
                    val lRowOffset = i*t
                    for(rRow in 0 until t){
                        val rightBlockIndex = rRow * m + offset
                        for(idx in 0 until blockSize) {
                            destArray[destBlockOffset + idx] = (destArray[destBlockOffset + idx] + leftArray[lRowOffset + rRow] * rightArray[rightBlockIndex + idx]).toPrimitive()
                        }
                    }
                }
                for(col in rMax until m) {
                    val destIdx = i*m + col
                    val lRow = i*t
                    for(rRow in 0 until t){
                        destArray[destIdx] = (destArray[destIdx] + leftArray[lRow + rRow] * rightArray[col + rRow * m]).toPrimitive()
                    }
                }

            }
        }

        return dest
    }

    companion object {

        suspend fun scalar(value: PrimitiveType): PrimitiveLNDArray {
            return PrimitiveLNDArray(PrimitiveArray(1) { value }, Strides.EMPTY)
        }

        suspend fun eyeLike(shape: IntArray, k: Int = 0): PrimitiveLNDArray {
            require(shape.size == 2) { "EyeLike is only supported for tensors of rank=2, current shape rank: ${shape.size}" }

            return PrimitiveLNDArray(shape) { it: IntArray ->
                val (row, column) = it
                if (column - k == row) PrimitiveConstants.ONE else PrimitiveConstants.ZERO
            }
        }


        @JvmName("invokeStrides")
        suspend operator fun invoke(strides: Strides): PrimitiveLNDArray {
            return PrimitiveLNDArray(PrimitiveArray(strides.linearSize), strides)
        }

        @JvmName("invokeStridesInlineInt")
        suspend operator fun invoke(strides: Strides, init: (Int) -> PrimitiveType): PrimitiveLNDArray {
            return PrimitiveLNDArray(PrimitiveArray(strides.linearSize, init), strides)
        }

        @JvmName("invokeStridesIntArray")
        suspend operator fun invoke(strides: Strides, init: (IntArray) -> PrimitiveType): PrimitiveLNDArray {
            val iterator = NDIndexer(strides)
            return PrimitiveLNDArray(strides) { _: Int -> init(iterator.next()) }
        }

        @JvmName("invokeShape")
        suspend operator fun invoke(shape: IntArray): PrimitiveLNDArray {
            val strides = Strides(shape)
            return PrimitiveLNDArray(PrimitiveArray(strides.linearSize), strides)
        }

        @JvmName("invokeShapeVarArg")
        suspend operator fun invoke(vararg shape: Int): PrimitiveLNDArray {
            val strides = Strides(shape)
            return PrimitiveLNDArray(PrimitiveArray(strides.linearSize), strides)
        }

        @JvmName("invokeShapeInlineInt")
        suspend operator fun invoke(shape: IntArray, init: (Int) -> PrimitiveType): PrimitiveLNDArray {
            val strides = Strides(shape)
            return PrimitiveLNDArray(PrimitiveArray(strides.linearSize, init), strides)
        }

        @JvmName("invokeShapeVarArgInlineInt")
        suspend operator fun invoke(vararg shape: Int, init: (Int) -> PrimitiveType): PrimitiveLNDArray {
            val strides = Strides(shape)
            return PrimitiveLNDArray(PrimitiveArray(strides.linearSize, init), strides)
        }

        @JvmName("invokeShapeIntArray")
        suspend operator fun invoke(shape: IntArray, init: (IntArray) -> PrimitiveType): PrimitiveLNDArray {
            return PrimitiveLNDArray(Strides(shape), init)
        }

        @JvmName("invokeShapeVarArgIntArray")
        suspend operator fun invoke(vararg shape: Int, init: (IntArray) -> PrimitiveType): PrimitiveLNDArray {
            return invoke(Strides(shape), init)
        }
    }

}
