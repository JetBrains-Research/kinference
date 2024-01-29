@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.arrays

import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlin.jvm.JvmName

@GenerateNameFromPrimitives
@MakePublic
internal open class MutablePrimitiveNDArray(array: PrimitiveTiledArray, strides: Strides = Strides.EMPTY) : PrimitiveNDArray(array, strides), MutableNumberNDArrayCore {
    constructor(shape: IntArray) : this(PrimitiveTiledArray(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> PrimitiveType) : this(PrimitiveTiledArray(shape, init), Strides(shape))

    constructor(strides: Strides) : this(PrimitiveTiledArray(strides), strides)
    constructor(strides: Strides, init: (Int) -> PrimitiveType) : this(PrimitiveTiledArray(strides, init), strides)

    override fun set(index: IntArray, value: Any) {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        array[linearIndex] = value as PrimitiveType
    }

    override fun setLinear(index: Int, value: Any) {
        array[index] = value as PrimitiveType
    }

    override fun viewMutable(vararg axes: Int): MutablePrimitiveNDArray {
        for ((i, axis) in axes.withIndex()) {
            require(shape[i] > axis)
        }

        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        if (array.blockSize == 0)
            return MutablePrimitiveNDArray(array, newStrides)

        val offsetBlocks = offset / array.blockSize
        val countBlocks = newStrides.linearSize / array.blockSize

//        val copyBlocks = array.copyOfRangeBlocks(offsetBlocks, offsetBlocks + countBlocks)
//        val newArray = PrimitiveTiledArray(copyBlocks)

        val newArray = if (newShape.isEmpty()) {
            val inBlockOffset = offset % array.blockSize
            this.array.view(offsetBlocks, countBlocks, inBlockOffset)
        } else {
            this.array.view(offsetBlocks, countBlocks)
        }

        return MutablePrimitiveNDArray(newArray, newStrides)
    }

    override fun copyIfNotMutable(): MutablePrimitiveNDArray {
        return MutablePrimitiveNDArray(array, strides)
    }

    override fun fill(value: Any, from: Int, to: Int) {
        value as PrimitiveType
        array.fill(value, from, to)
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        array as PrimitiveNDArray
        this.array.fill(array.array[index], from, to)
    }

    override suspend fun mapMutable(function: PrimitiveToPrimitiveFunction): MutablePrimitiveNDArray {
        function as PrimitiveMap

        for (blockIdx in array.indices) {
            val block = array.getBlock(blockIdx)
            for (idx in block.indices) {
                block[idx] = function.apply(block[idx])
            }
        }

        return this
    }

    override suspend operator fun plusAssign(other: NumberNDArray) {
        other as PrimitiveNDArray

        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] + other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[0]
                for (blockIdx in this.array.indices) {
                    val block = this.array.getBlock(blockIdx)
                    for (idx in block.indices) {
                        block[idx] = (block[idx] + scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Plus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                // TODO change to real plusAssign
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in left.array.indices) {
                    val leftBlock = left.array.getBlock(blockNum)
                    val rightBlock = right.array.getBlock(blockNum)
                    val destBlock = dest.array.getBlock(blockNum)

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] + rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }
    }

    override suspend operator fun minusAssign(other: NumberNDArray) {
        other as PrimitiveNDArray

        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] - other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[0]
                for (blockIdx in this.array.indices) {
                    val block = this.array.getBlock(blockIdx)
                    for (idx in block.indices) {
                        block[idx] = (block[idx] - scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Minus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in left.array.indices) {
                    val leftBlock = left.array.getBlock(blockNum)
                    val rightBlock = right.array.getBlock(blockNum)
                    val destBlock = dest.array.getBlock(blockNum)

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] - rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }
    }

    override suspend operator fun timesAssign(other: NumberNDArray) {
        other as PrimitiveNDArray

        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] * other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[0]
                for (blockIdx in this.array.indices) {
                    val block = this.array.getBlock(blockIdx)
                    for (idx in block.indices) {
                        block[idx] = (block[idx] * scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Times assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.getBlock(blockNum)
                    val rightBlock = right.array.getBlock(blockNum)
                    val destBlock = dest.array.getBlock(blockNum)

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] * rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }
    }

    override suspend operator fun divAssign(other: NumberNDArray) {
        other as PrimitiveNDArray

        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] / other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[0]
                for (blockIdx in this.array.indices) {
                    val block = this.array.getBlock(blockIdx)
                    for (idx in block.indices) {
                        block[idx] = (block[idx] / scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Div assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in left.array.indices) {
                    val leftBlock = left.array.getBlock(blockNum)
                    val rightBlock = right.array.getBlock(blockNum)
                    val destBlock = dest.array.getBlock(blockNum)

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] / rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as PrimitiveNDArray
        other.array.copyInto(this.array, offset, startInOther, endInOther)
    }

    override fun clean() {
        for (blockIdx in array.indices) {
            array.getBlock(blockIdx).fill(PrimitiveConstants.ZERO)
        }
    }

    companion object {
        fun scalar(value: PrimitiveType): MutablePrimitiveNDArray {
            return MutablePrimitiveNDArray(PrimitiveTiledArray(1, 1) { value }, Strides.EMPTY)
        }

        operator fun invoke(strides: Strides, init: (IntArray) -> PrimitiveType): MutablePrimitiveNDArray {
            val iterator = NDIndexer(strides)
            return MutablePrimitiveNDArray(strides) { init(iterator.next()) }
        }

        operator fun invoke(shape: IntArray, init: (IntArray) -> PrimitiveType): MutablePrimitiveNDArray {
            return invoke(Strides(shape), init)
        }

        operator fun invoke(vararg shape: Int): MutablePrimitiveNDArray {
            return MutablePrimitiveNDArray(PrimitiveTiledArray(shape), Strides(shape))
        }

        @JvmName("invokeNDVarArg")
        operator fun invoke(vararg shape: Int, init: (IntArray) -> PrimitiveType): MutablePrimitiveNDArray {
            return invoke(Strides(shape), init)
        }

        @JvmName("invokeVarArg")
        operator fun invoke(vararg shape: Int, init: (Int) -> PrimitiveType): MutablePrimitiveNDArray {
            return MutablePrimitiveNDArray(PrimitiveTiledArray(shape, init), Strides(shape))
        }
    }
}
