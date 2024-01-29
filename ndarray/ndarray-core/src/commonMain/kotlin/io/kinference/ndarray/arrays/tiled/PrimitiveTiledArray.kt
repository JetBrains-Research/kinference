@file:GeneratePrimitives(DataType.ALL)
@file:Suppress("DuplicatedCode")

package io.kinference.ndarray.arrays.tiled

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.blockSizeByStrides
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlin.math.min

@GenerateNameFromPrimitives
@MakePublic
internal class PrimitiveTiledArray {
    val size: Int
    val blockSize: Int
    val blocksNum: Int
    private val blocks: Array<PrimitiveArray>
    private val marker: Array<StateMarker>

    private val blocksOffset: Int
    private val inBlockIdx: Int?

    val indices: IntRange

    companion object {
        private val type: ArrayTypes = ArrayTypes.valueOf(PrimitiveArray::class.simpleName!!)
        private val emptyMarker: Array<StateMarker> = arrayOf()

        operator fun invoke(strides: Strides): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            return PrimitiveTiledArray(strides.linearSize, blockSize)
        }

        operator fun invoke(strides: Strides, init: (Int) -> PrimitiveType): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            return PrimitiveTiledArray(strides.linearSize, blockSize, init)
        }

        operator fun invoke(shape: IntArray) = invoke(Strides(shape))

        operator fun invoke(shape: IntArray, init: (Int) -> PrimitiveType) = invoke(Strides(shape), init)

        operator fun invoke(strides: Strides, array: PrimitiveArray): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            val countBlocks = array.size / blockSize
            val blocksArray = Array(countBlocks) { PrimitiveArray(blockSize) }
            repeat(countBlocks) { blockNum ->
                array.copyInto(blocksArray[blockNum], startIndex = blockNum * blockSize, endIndex = (blockNum + 1) * blockSize)
            }

            return PrimitiveTiledArray(blocksArray)
        }

        fun matrixLike(shape: IntArray, init: (Int) -> PrimitiveType): PrimitiveTiledArray {
            require(shape.size == 1 || shape.size == 2) { "NDArray should be of rank <= 2. Got rank=${shape.size}" }

            var count = 0
            val blockSize = shape.last()
            val blocksNum = if (shape.size == 1) 1 else shape[0]
            val blocks = Array(blocksNum) { PrimitiveArray(blockSize) }
            for (block in blocks) {
                for (idx in 0 until blockSize) {
                    block[idx] = init(count++)
                }
            }
            return PrimitiveTiledArray(blocks)
        }
    }

    constructor(size: Int, blockSize: Int) {
        if (blockSize != 0)
            require(size % blockSize == 0) { "Size must divide blockSize" }

        this.blocksNum = if (blockSize == 0) 0 else size / blockSize
        this.blockSize = blockSize
        this.size = size

        // With array dispatcher
        val containerArray = ArraysDispatcher.getArraysAndMarkers<PrimitiveArray>(type, this.blockSize, this.blocksNum)
        this.blocks = Array(containerArray.size) { i -> containerArray[i].array }
        this.marker = Array(containerArray.size) { i -> containerArray[i].markAsOutput }
        this.indices = 0 until blocksNum
        // Without memory management
//        this.blocks = Array(blocksNum) { PrimitiveArray(blockSize) }
//        this.marker = emptyMarker

        this.blocksOffset = 0
//        this.endBlockIdx = blocks.size
        this.inBlockIdx = null
    }

    constructor(
        blocks: Array<PrimitiveArray>,
        markers: Array<(ArrayUsageMarker) -> Unit> = emptyMarker,
        blocksOffset: Int = 0,
        blocksCount: Int = blocks.size,
        inBlockIdx: Int? = null
    ) {
        require(blocksOffset >= 0 && blocksCount > 0 && blocksOffset + blocksCount <= blocks.size)

        this.blocks = blocks
        this.blocksNum = blocksCount
        this.blockSize = if (blocks.isEmpty()) 0 else blocks.first().size

        this.blocksOffset = blocksOffset
//        this.endBlockIdx = endBlockIdx
        if (inBlockIdx != null) require(blocksNum == 1 && inBlockIdx < blockSize)
        this.inBlockIdx = inBlockIdx

//        this.blocksNum = blocks.size
        this.size = if (inBlockIdx == null) this.blocksNum * this.blockSize else 1
        this.marker = markers
        this.indices = 0 until blocksNum
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

        for (blockIdx in indices) {
            val block = blocks[blocksOffset + blockIdx]
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
        require(i < size)

        val inBlockOffset = inBlockIdx ?: 0

        val (blockIdx, blockOff) = indexFor(i)
        return blocks[blocksOffset + blockIdx][inBlockOffset + blockOff]
    }

    operator fun set(i: Int, value: PrimitiveType) {
        require(i < size)

        val inBlockOffset = inBlockIdx ?: 0


        val (blockIdx, blockOff) = indexFor(i)
        blocks[blocksOffset + blockIdx][inBlockOffset + blockOff] = value
    }

    fun copyOf(): PrimitiveTiledArray {
        require(inBlockIdx == null)
        val copyArray = PrimitiveTiledArray(size, blockSize)

        for (blockNum in 0 until blocksNum) {
            val thisBlock = this.blocks[blocksOffset + blockNum]
            val destBlock = copyArray.blocks[blockNum]

            thisBlock.copyInto(destBlock)
        }

        return copyArray
    }

    fun copyInto(dest: PrimitiveTiledArray, destOffset: Int = 0, srcStart: Int = 0, srcEnd: Int = size) {
        require(srcStart >= 0 && srcEnd <= size && srcStart <= srcEnd && srcEnd - srcStart <= dest.size - destOffset)


        if (srcStart == srcEnd)
            return

        val thisPtr = this.pointer(srcStart) //PrimitivePointer(this, srcStart)
        val destPtr = dest.pointer(destOffset) //PrimitivePointer(dest, destOffset)

        destPtr.accept(thisPtr, srcEnd - srcStart) { _: PrimitiveType, src: PrimitiveType -> src }
    }

    fun copyOfRange(fromIndex: Int, toIndex: Int): PrimitiveArray {
        require(fromIndex >= 0 && toIndex <= size && fromIndex <= toIndex)

        val array = PrimitiveArray(toIndex - fromIndex)
        val pointer = this.pointer(fromIndex) //PrimitivePointer(this, fromIndex)

        for (i in array.indices) {
            array[i] = pointer.getAndIncrement()
        }

        return array
    }

    fun fill(value: PrimitiveType, from: Int = 0, to: Int = size) {
        require(from >= 0 && to <= size && from <= to)
        if (from == to)
            return

        val pointer = this.pointer(from) //PrimitivePointer(this, from)

        var count = to - from

        while (count > 0) {
            val block = pointer.currentBlock
            val offset = pointer.indexInBlock
            pointer.blockIncrement()

            block.fill(value, offset, min(blockSize, count + offset))

            count -= blockSize
        }
    }

    fun getBlock(index: Int): PrimitiveArray {
        require(inBlockIdx == null && index < blocksNum)

        return blocks[blocksOffset + index]
    }

    fun setBlock(index: Int, block: PrimitiveArray) {
        require(inBlockIdx == null && index < blocksNum && block.size == this.blockSize)

        blocks[blocksOffset + index] = block
    }

    fun getMarker(index: Int): StateMarker {
        require(inBlockIdx == null && index < blocksNum)

        return marker[blocksOffset + index]
    }

    fun copyOfBlocks(): Array<PrimitiveArray> {
        require(inBlockIdx == null)
        return blocks.copyOfRange(blocksOffset, blocksOffset + blocksNum)
    }

    fun copyOfMarkers(): Array<StateMarker> {
        require(inBlockIdx == null)

        return marker.copyOfRange(blocksOffset, blocksOffset + blocksNum)
    }

    fun copyOfRangeBlocks(fromIndex: Int, toIndex: Int): Array<PrimitiveArray>  {
        require(inBlockIdx == null && fromIndex >= 0 && toIndex <= blocksNum && fromIndex < toIndex)

        return blocks.copyOfRange(blocksOffset + fromIndex, blocksOffset + toIndex)
    }

    fun copyIntoBlocks(destination: Array<PrimitiveArray>, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = blocksNum) {
        require(inBlockIdx == null && startIndex >= 0 && endIndex <= blocksNum && startIndex < endIndex && endIndex - startIndex <= destination.size - destinationOffset)

        blocks.copyInto(destination, destinationOffset, blocksOffset + startIndex, blocksOffset + endIndex)
    }

    fun copyIntoMarkers(destination: Array<StateMarker>, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = blocksNum) {
        require(inBlockIdx == null && startIndex >= 0 && endIndex <= blocksNum && startIndex < endIndex && endIndex - startIndex <= destination.size - destinationOffset)

        marker.copyInto(destination, destinationOffset, blocksOffset + startIndex, blocksOffset + endIndex)
    }

    fun view(blocksOffset: Int, blocksCount: Int, inBlockIdx: Int? = null): PrimitiveTiledArray {
        require(blocksOffset >= 0 && blocksCount > 0 && this.blocksOffset + blocksOffset + blocksCount <= blocks.size)

        return PrimitiveTiledArray(blocks, marker, this.blocksOffset + blocksOffset, blocksCount, inBlockIdx)
    }
}
