package io.kinference.ndarray.arrays.tiled

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.pointers.BooleanPointer
import io.kinference.ndarray.arrays.pointers.accept
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import kotlin.math.min

class BooleanTiledArray {
    val size: Int
    val blockSize: Int
    val blocksNum: Int
    val blocks: Array<BooleanArray>

    companion object {
        const val MIN_BLOCK_SIZE = 512
        val logger: Logger = LoggerFactory.getLogger(LongTiledArray::class.java)

        private fun blockSizeByStrides(strides: Strides, divider: Int = 1): Int {
            return when {
                strides.linearSize == 0 -> 0
                strides.shape.isEmpty() -> 1
                else -> {
                    val rowSize = strides.shape.last()

                    require(rowSize % divider == 0)

                    val dividedRowSize = rowSize / divider

                    val blockSize = if (dividedRowSize < MIN_BLOCK_SIZE) dividedRowSize else {
                        var num = dividedRowSize / MIN_BLOCK_SIZE
                        while (dividedRowSize % num != 0) num--
                        dividedRowSize / num
                    }

                    blockSize
                }
            }
        }

        operator fun invoke(strides: Strides, divider: Int = 1): BooleanTiledArray {
            val blockSize = blockSizeByStrides(strides, divider)
            return BooleanTiledArray(strides.linearSize, blockSize)
        }

        operator fun invoke(array: BooleanArray, strides: Strides, divider: Int = 1): BooleanTiledArray {
            require(strides.linearSize == array.size)

            val blockSize = blockSizeByStrides(strides, divider)
            return BooleanTiledArray(array, blockSize)
        }

        operator fun invoke(strides: Strides, divider: Int = 1, init: (Int) -> Boolean): BooleanTiledArray {
            val blockSize = blockSizeByStrides(strides, divider)
            return BooleanTiledArray(strides.linearSize, blockSize, init)
        }

        operator fun invoke(shape: IntArray, divider: Int = 1) = invoke(Strides(shape), divider)

        operator fun invoke(array: BooleanArray, shape: IntArray, divider: Int = 1) = invoke(array, Strides(shape), divider)

        operator fun invoke(shape: IntArray, divider: Int = 1, init: (Int) -> Boolean) = invoke(Strides(shape), divider, init)
    }

    constructor(size: Int, blockSize: Int) {
        if (blockSize != 0)
            require(size % blockSize == 0) { "Size must divide blockSize" }

        this.blocksNum = if (blockSize == 0) 0 else size / blockSize
        this.blocks = Array(blocksNum) { BooleanArray(blockSize) }
        this.blockSize = blockSize
        this.size = size
    }

    constructor(blocks: Array<BooleanArray>) {
        this.blocks = blocks
        this.blockSize = if (blocks.isEmpty()) 0 else blocks.first().size
        this.blocksNum = blocks.size
        this.size = this.blocksNum * this.blockSize
    }

    constructor(size: Int, blockSize: Int, init: (Int) -> Boolean) : this(size, blockSize) {
        var count = 0
        for (block in blocks) {
            for (idx in 0 until blockSize) {
                block[idx] = init(count++)
            }
        }
    }

    constructor(array: BooleanArray, blockSize: Int) : this(array.size, blockSize) {
        var startIndex = 0
        var endIndex = blockSize

        for (block in blocks) {
            array.copyInto(block, 0, startIndex, endIndex)
            startIndex = endIndex
            endIndex += blockSize
        }
    }

    fun pointer(startIndex: Int = 0) = BooleanPointer(this, startIndex)

    fun toArray(): BooleanArray {
        if (size == 0) {
            return BooleanArray(0)
        }

        val array = BooleanArray(size)
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

    operator fun get(i: Int): Boolean {
        val (blockIdx, blockOff) = indexFor(i)
        return blocks[blockIdx][blockOff]
    }

    operator fun set(i: Int, value: Boolean) {
        val (blockIdx, blockOff) = indexFor(i)
        blocks[blockIdx][blockOff] = value
    }

    fun copyOf(): BooleanTiledArray {
        val copyArray = BooleanTiledArray(size, blockSize)

        for (blockNum in 0 until blocksNum) {
            val thisBlock = this.blocks[blockNum]
            val destBlock = copyArray.blocks[blockNum]

            thisBlock.copyInto(destBlock)
        }

        return copyArray
    }

    fun copyInto(dest: BooleanTiledArray, destOffset: Int = 0, srcStart: Int = 0, srcEnd: Int = size) {
        if (srcStart == srcEnd)
            return

        val thisPtr = BooleanPointer(this, srcStart)
        val destPtr = BooleanPointer(dest, destOffset)

        destPtr.accept(thisPtr, srcEnd - srcStart) { dst, src -> src }
    }

    fun copyOfRange(fromIndex: Int, toIndex: Int): BooleanArray {
        val array = BooleanArray(toIndex - fromIndex)
        val pointer = BooleanPointer(this, fromIndex)

        for (i in array.indices) {
            array[i] = pointer.getAndIncrement()
        }

        return array
    }

    fun fill(value: Boolean, from: Int = 0, to: Int = size) {
        if (from == to)
            return

        val pointer = BooleanPointer(this, from)

        var count = to - from

        while (count > 0) {
            val (block, offset) = pointer.getAndIncrementBlock()
            block.fill(value, offset, min(blockSize, count))

            count -= blockSize
        }
    }
}
