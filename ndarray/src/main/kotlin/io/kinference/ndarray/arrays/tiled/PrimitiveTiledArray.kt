@file:GenerateWithPrimitives

package io.kinference.ndarray.arrays.tiled

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.primitives.annotations.GenerateWithPrimitives
import io.kinference.primitives.annotations.PrimitiveClass
import io.kinference.primitives.types.*
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import kotlin.math.min

@PrimitiveClass
class PrimitiveTiledArray {
    val size: Int
    val blockSize: Int
    val blocksNum: Int
    val blocks: Array<PrimitiveArray>

    companion object {
        const val MIN_BLOCK_SIZE = 64
        val logger: Logger = LoggerFactory.getLogger(PrimitiveTiledArray::class.java)

        operator fun invoke(strides: Strides): PrimitiveTiledArray {
            return when {
                strides.linearSize == 0 -> PrimitiveTiledArray(0, 0)
                strides.shape.isEmpty() -> PrimitiveTiledArray(1, 1)
                else -> {
                    val rowSize = strides.shape.last()
                    val blockSize = if (rowSize < MIN_BLOCK_SIZE) rowSize else {
                        var num = rowSize / MIN_BLOCK_SIZE
                        while (rowSize % num != 0) num--
                        rowSize / num
                    }

                    PrimitiveTiledArray(strides.linearSize, blockSize)
                }
            }
        }

        operator fun invoke(array: PrimitiveArray, strides: Strides): PrimitiveTiledArray {
            require(strides.linearSize == array.size)

            val tiledArray = PrimitiveTiledArray(strides)

            var startIndex = 0
            var endIndex = tiledArray.blockSize
            for (block in tiledArray.blocks) {
                array.copyInto(block, 0, startIndex, endIndex)
                startIndex = endIndex
                endIndex += tiledArray.blockSize
            }

            return tiledArray
        }

        operator fun invoke(strides: Strides, init: (Int) -> PrimitiveType): PrimitiveTiledArray {
            val tiledArray = PrimitiveTiledArray(strides)

            var count = 0
            for (block in tiledArray.blocks) {
                for (idx in 0 until tiledArray.blockSize) {
                    block[idx] = init(count++)
                }
            }

            return tiledArray
        }

        operator fun invoke(shape: IntArray) = invoke(Strides(shape))

        operator fun invoke(array: PrimitiveArray, shape: IntArray) = invoke(array, Strides(shape))

        operator fun invoke(shape: IntArray, init: (Int) -> PrimitiveType) = invoke(Strides(shape), init)
    }

    constructor(size: Int, blockSize: Int) {
        if (blockSize != 0)
            require(size % blockSize == 0) { "Size must divide blockSize" }

        this.blocksNum = if (blockSize == 0) 0 else size / blockSize
        this.blocks = Array(blocksNum) { PrimitiveArray(blockSize) }
        this.blockSize = blockSize
        this.size = size
    }

    constructor(blocks: Array<PrimitiveArray>) {
        this.blocks = blocks
        this.blockSize = if (blocks.isEmpty()) 0 else blocks.first().size
        this.blocksNum = blocks.size
        this.size = this.blocksNum * this.blockSize
    }

    constructor(size: Int, blockSize: Int, init: (Int) -> PrimitiveType) : this(size, blockSize) {
        var count = 0
        for (block in blocks) {
            for (idx in 0 until blockSize) {
                block[idx] = init(count++)
            }
        }
    }

    fun pointer(startIndex: Int = 0) = PrimitivePointer(this, startIndex)

    fun toArray(): PrimitiveArray {
        if (size == 0) {
            return PrimitiveArray(0)
        }

        val array = PrimitiveArray(size)
        var offset = 0

        for (block in blocks) {
            block.copyInto(array, offset)
            offset += blockSize
        }

        return array
    }

    fun indexFor(i: Int): Pair<Int, Int> {
        val blockIdx = i / blockSize
        val blockOff = i % blockSize
        return blockIdx to blockOff
    }

    operator fun get(i: Int): PrimitiveType {
        val (blockIdx, blockOff) = indexFor(i)
        return blocks[blockIdx][blockOff]
    }

    operator fun set(i: Int, value: PrimitiveType) {
        val (blockIdx, blockOff) = indexFor(i)
        blocks[blockIdx][blockOff] = value
    }

    fun copyOf(): PrimitiveTiledArray {
        val copyArray = PrimitiveTiledArray(size, blockSize)

        for (blockNum in 0 until blocksNum) {
            val thisBlock = this.blocks[blockNum]
            val destBlock = copyArray.blocks[blockNum]

            thisBlock.copyInto(destBlock)
        }

        return copyArray
    }

    fun copyInto(dest: PrimitiveTiledArray, destOffset: Int = 0, srcStart: Int = 0, srcEnd: Int = size) {
        if (srcStart == srcEnd)
            return

        val thisPtr = PrimitivePointer(this, srcStart)
        val destPtr = PrimitivePointer(dest, destOffset)

        destPtr.accept(thisPtr, srcEnd - srcStart) { dst, src -> src }
    }

    fun plus(other: PrimitiveTiledArray): PrimitiveTiledArray {
        val thisPtr = PrimitivePointer(this)
        val destPtr = PrimitivePointer(other)
        thisPtr.accept(destPtr, this.size) { src, dst -> (src + dst).toPrimitive() }
        return this
    }

    fun copyOfRange(fromIndex: Int, toIndex: Int): PrimitiveArray {
        val array = PrimitiveArray(toIndex - fromIndex)
        val pointer = PrimitivePointer(this, fromIndex)

        for (i in array.indices) {
            array[i] = pointer.getAndIncrement()
        }

        return array
    }

    fun fill(value: PrimitiveType, from: Int = 0, to: Int = size) {
        if (from == to)
            return

        val pointer = PrimitivePointer(this, from)

        var count = to - from

        while (count > 0) {
            val (block, offset) = pointer.getAndIncrementBlock()
            block.fill(value, offset, min(blockSize, count))

            count -= blockSize
        }
    }
}
