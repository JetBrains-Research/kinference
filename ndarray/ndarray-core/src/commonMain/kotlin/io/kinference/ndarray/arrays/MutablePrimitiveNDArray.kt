@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.arrays

import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import io.kinference.utils.InlineInt
import kotlin.jvm.JvmName

@GenerateNameFromPrimitives
@MakePublic
internal open class MutablePrimitiveNDArray(array: PrimitiveTiledArray, strides: Strides = Strides.EMPTY) : PrimitiveNDArray(array, strides), MutableNumberNDArrayCore {
    constructor(shape: IntArray) : this(PrimitiveTiledArray(shape), Strides(shape))
    constructor(shape: IntArray, init: (InlineInt) -> PrimitiveType) : this(PrimitiveTiledArray(shape, init), Strides(shape))

    constructor(strides: Strides) : this(PrimitiveTiledArray(strides), strides)
    constructor(strides: Strides, init: (InlineInt) -> PrimitiveType) : this(PrimitiveTiledArray(strides, init), strides)

    override fun set(index: IntArray, value: Any) {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        array[linearIndex] = value as PrimitiveType
    }

    override fun setLinear(index: Int, value: Any) {
        array[index] = value as PrimitiveType
    }

    override fun viewMutable(vararg axes: Int): MutablePrimitiveNDArray {
        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }
        val offsetBlocks = offset / array.blockSize

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = PrimitiveTiledArray(copyBlocks)

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
        val (blockIndex, blockOffset) = array.array.indexFor(index)
        this.array.fill(array.array.blocks[blockIndex][blockOffset], from, to)
    }

    override suspend fun mapMutable(function: PrimitiveToPrimitiveFunction): MutablePrimitiveNDArray {
        function as PrimitiveMap

        for (block in array.blocks) {
            for (idx in block.indices) {
                block[idx] = function.apply(block[idx])
            }
        }

        return this
    }

    override suspend operator fun plusAssign(other: NumberNDArray) {
        other as PrimitiveNDArray

        when {
            this.isScalar() && other.isScalar() -> this.array.blocks[0][0] = (this.array.blocks[0][0] + other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array.blocks[0][0]
                for (block in this.array.blocks) {
                    for (idx in block.indices) {
                        block[idx] = (block[idx] + scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Plus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                // TODO change to real plusAssign
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                /*val leftArray = left.array.toArray()
                val rightArray = right.array.toArray()
                val destArray = dest.array.toArray()

                for (index in 0 until left.linearSize) {
                    destArray[index] = (leftArray[index] + rightArray[index]).toPrimitive()
                }

                dest.array = PrimitiveTiledArray(destArray, dest.strides)*/

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

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
            this.isScalar() && other.isScalar() -> this.array.blocks[0][0] = (this.array.blocks[0][0] - other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array.blocks[0][0]
                for (block in this.array.blocks) {
                    for (idx in block.indices) {
                        block[idx] = (block[idx] - scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Minus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

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
            this.isScalar() && other.isScalar() -> this.array.blocks[0][0] = (this.array.blocks[0][0] * other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array.blocks[0][0]
                for (block in this.array.blocks) {
                    for (idx in block.indices) {
                        block[idx] = (block[idx] * scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Times assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

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
            this.isScalar() && other.isScalar() -> this.array.blocks[0][0] = (this.array.blocks[0][0] / other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array.blocks[0][0]
                for (block in this.array.blocks) {
                    for (idx in block.indices) {
                        block[idx] = (block[idx] / scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Div assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

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
        for (block in array.blocks) {
            block.fill((0).toPrimitive())
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
        operator fun invoke(vararg shape: Int, init: (InlineInt) -> PrimitiveType): MutablePrimitiveNDArray {
            return MutablePrimitiveNDArray(PrimitiveTiledArray(shape, init), Strides(shape))
        }
    }
}
