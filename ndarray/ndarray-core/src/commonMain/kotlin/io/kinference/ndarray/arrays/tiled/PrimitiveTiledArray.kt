@file:GeneratePrimitives(DataType.ALL)
@file:Suppress("DuplicatedCode")

package io.kinference.ndarray.arrays.tiled

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.ArrayDispatcher
import io.kinference.ndarray.arrays.memory.PrimitiveArrayContainer
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.blockSizeByStrides
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import io.kinference.ndarray.inlines.*
import io.kinference.utils.ModelContext
import kotlin.coroutines.coroutineContext
import kotlin.math.min

@GenerateNameFromPrimitives
@MakePublic
internal class PrimitiveTiledArray(val blocks: Array<PrimitiveArray>, val marker: Array<StateMarker> = emptyMarker) {
    val size: Int
    val blockSize: Int = if (blocks.isEmpty()) 0 else blocks.first().size
    val blocksNum: Int = blocks.size

    init {
        this.size = this.blocksNum * this.blockSize
    }

    companion object {
        val type: ArrayTypes = ArrayTypes.valueOf(PrimitiveArray::class.simpleName!!)
        private val emptyMarker: Array<StateMarker> = arrayOf()

        suspend operator fun invoke(strides: Strides): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            return PrimitiveTiledArray(strides.linearSize, blockSize)
        }

        suspend operator fun invoke(strides: Strides, init: (InlineInt) -> PrimitiveType): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            return PrimitiveTiledArray(strides.linearSize, blockSize, init)
        }

        suspend operator fun invoke(shape: IntArray) = invoke(Strides(shape))

        suspend operator fun invoke(shape: IntArray, init: (InlineInt) -> PrimitiveType) = invoke(Strides(shape), init)

        operator fun invoke(strides: Strides, array: PrimitiveArray): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            val countBlocks = array.size / blockSize
            val blocksArray = Array(countBlocks) { PrimitiveArray(blockSize) }
            repeat(countBlocks) { blockNum ->
                array.copyInto(blocksArray[blockNum], startIndex = blockNum * blockSize, endIndex = (blockNum + 1) * blockSize)
            }

            return PrimitiveTiledArray(blocksArray)
        }

        suspend operator fun invoke(size: Int, blockSize: Int): PrimitiveTiledArray {
            if (blockSize != 0)
                require(size % blockSize == 0) { "Size must divide blockSize" }

            val blocksNum = if (blockSize == 0) 0 else size / blockSize

            val modelName = coroutineContext[ModelContext.Key]?.modelName ?: NO_CONTEXT

            // With array dispatcher
            val containerArray = ArrayDispatcher.getArraysAndMarkers(modelName, type, blockSize, blocksNum)
            val blocks = Array(containerArray.size) { i -> (containerArray[i] as PrimitiveArrayContainer).array }
            val marker = Array(containerArray.size) { i -> containerArray[i].markAsOutput }

            return PrimitiveTiledArray(blocks, marker)
        }

        suspend operator fun invoke(size: Int, blockSize: Int, init: (InlineInt) -> PrimitiveType) : PrimitiveTiledArray {
            val tiledArray = PrimitiveTiledArray(size, blockSize)
            var count = 0
            for (block in tiledArray.blocks) {
                for (idx in 0 until blockSize) {
                    block[idx] = init(InlineInt(count++))
                }
            }

            return tiledArray
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

    suspend fun copyOf(): PrimitiveTiledArray {
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

        destPtr.accept(thisPtr, srcEnd - srcStart) { _: PrimitiveType, src: PrimitiveType -> src }
    }

    @FilterPrimitives(exclude = [DataType.BOOLEAN])
    fun plus(other: PrimitiveTiledArray): PrimitiveTiledArray {
        val thisPtr = PrimitivePointer(this)
        val destPtr = PrimitivePointer(other)
        thisPtr.accept(destPtr, this.size) { src: PrimitiveType, dst: PrimitiveType -> (src + dst).toPrimitive() }
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
            val block = pointer.currentBlock
            val offset = pointer.indexInBlock
            pointer.blockIncrement()

            block.fill(value, offset, min(blockSize, count + offset))

            count -= blockSize
        }
    }
}
