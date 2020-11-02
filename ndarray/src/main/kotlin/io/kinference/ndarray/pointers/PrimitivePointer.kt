@file:GenerateWithPrimitives
@file:Suppress("NOTHING_TO_INLINE", "EXPERIMENTAL_API_USAGE", "DuplicatedCode")

package io.kinference.ndarray.pointers


import io.kinference.ndarray.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.GenerateWithPrimitives
import io.kinference.primitives.annotations.PrimitiveBinding
import io.kinference.primitives.annotations.PrimitiveClass
import io.kinference.primitives.annotations.Type1
import io.kinference.primitives.types.*
import kotlin.math.min

@PrimitiveClass
class PrimitivePointer {
    data class BlockWithOffset(val block: PrimitiveArray, val offset: Int)

    val array: PrimitiveTiledArray

    var blockNum: Int
    var indexInBlock: Int

    var currentBlock: PrimitiveArray

    constructor(array: PrimitiveTiledArray, startIndex: Int = 0) {
        require(startIndex >= 0 && startIndex < array.size) { "Start index of Iterator must be >= 0 and < array size" }
        this.array = array
        this.blockNum = startIndex / array.blockSize
        this.indexInBlock = startIndex % array.blockSize
        this.currentBlock = array.blocks[blockNum]
    }

    constructor(other: PrimitivePointer) {
        this.array = other.array
        this.blockNum = other.blockNum
        this.indexInBlock = other.indexInBlock
        this.currentBlock = other.currentBlock
    }

    var linearIndex: Int
        get() = blockNum * array.blockSize + indexInBlock
        set(value) {
            require(value >= 0 && value < array.size) { "Linear index of Iterator must be >= 0 and < array size" }
            this.blockNum = value / array.blockSize
            this.indexInBlock = value % array.blockSize
            this.currentBlock = array.blocks[blockNum]
        }

    inline fun set(value: PrimitiveType) {
        currentBlock[indexInBlock] = value
    }

    inline fun get(): PrimitiveType {
        return currentBlock[indexInBlock]
    }

    inline fun blockIncrement() {
        when {
            blockNum < array.blocksNum - 1 -> {
                blockNum++
                indexInBlock = 0
                currentBlock = array.blocks[blockNum]
            }
            else -> indexInBlock++
        }
    }

    inline fun increment() {
        when {
            indexInBlock < array.blockSize - 1 -> indexInBlock++
            else -> blockIncrement()
        }
    }

    inline fun incrementAndGet(): PrimitiveType {
        increment()
        return currentBlock[indexInBlock]
    }

    inline fun getAndIncrement(): PrimitiveType {
        val value = currentBlock[indexInBlock]
        increment()
        return value
    }

    inline fun incrementAndGetBlock(): BlockWithOffset {
        blockIncrement()
        return BlockWithOffset(currentBlock, indexInBlock)
    }

    inline fun getAndIncrementBlock(): BlockWithOffset {
        val value = BlockWithOffset(currentBlock, indexInBlock)
        blockIncrement()
        return value
    }

    fun isValid(): Boolean = indexInBlock < array.blockSize
}

@PrimitiveBinding(type1 = [DataType.BYTE, DataType.SHORT, DataType.INT, DataType.LONG,
    DataType.UBYTE, DataType.USHORT, DataType.UINT, DataType.ULONG, DataType.FLOAT, DataType.DOUBLE])
inline fun PrimitivePointer.isCompatibleWith(other: @Type1 PrimitivePointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

@PrimitiveBinding(type1 = [DataType.BYTE, DataType.SHORT, DataType.INT, DataType.LONG,
    DataType.UBYTE, DataType.USHORT, DataType.UINT, DataType.ULONG, DataType.FLOAT, DataType.DOUBLE])
inline fun PrimitivePointer.isCompatibleBySize(other: @Type1 PrimitivePointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun PrimitivePointer.map(count: Int, action: (value: PrimitiveType) -> PrimitiveType) {
    var end = count
    while (end > 0) {
        val (block, offset) = this.getAndIncrementBlock()

        for (index in offset until min(block.size, offset + end)) {
            block[index] = action(block[index])
        }

        end -= block.size
    }
}

@PrimitiveBinding(type1 = [DataType.BYTE, DataType.SHORT, DataType.INT, DataType.LONG,
    DataType.UBYTE, DataType.USHORT, DataType.UINT, DataType.ULONG, DataType.FLOAT, DataType.DOUBLE])
inline fun PrimitivePointer.mapTo(container: @Type1 PrimitivePointer, count: Int, action: (value: PrimitiveType) -> @Type1 PrimitiveType) {
    require(this.isCompatibleBySize(container, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(container)) {
        while (end > 0) {
            val (srcBlock, offset) = this.getAndIncrementBlock()
            val (dstBlock, _) = container.getAndIncrementBlock()

            for (index in offset until min(srcBlock.size, offset + end)) {
                dstBlock[index] = action(srcBlock[index])
            }

            end -= srcBlock.size
        }
    } else {
        while (end > 0) {
            container.set(action(this.getAndIncrement()))
            container.increment()
            end--
        }
    }
}

@PrimitiveBinding(type1 = [DataType.BYTE, DataType.SHORT, DataType.INT, DataType.LONG,
    DataType.UBYTE, DataType.USHORT, DataType.UINT, DataType.ULONG, DataType.FLOAT, DataType.DOUBLE])
inline fun PrimitivePointer.accept(other: @Type1 PrimitivePointer, count: Int, action: (dst: PrimitiveType, src: @Type1 PrimitiveType) -> PrimitiveType) {
    require(this.isCompatibleBySize(other, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(other)) {
        while (end > 0) {
            val (dstBlock, dstOffset) = this.getAndIncrementBlock()
            val (srcBlock, _) = other.getAndIncrementBlock()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], srcBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(this.get(), other.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

inline fun PrimitivePointer.acceptDouble(first: PrimitivePointer, second: PrimitivePointer, count: Int, action: (dst: PrimitiveType, fst: PrimitiveType, snd: PrimitiveType) -> PrimitiveType) {
    require(this.isCompatibleBySize(first, count)) { "Pointers not compatible by available elements" }
    require(this.isCompatibleBySize(second, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(first) && this.isCompatibleWith(second)) {
        while (end > 0) {
            val (dstBlock, dstOffset) = this.getAndIncrementBlock()
            val (fstBlock, _) = first.getAndIncrementBlock()
            val (sndBlock, _) = second.getAndIncrementBlock()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(this.get(), first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

inline fun PrimitivePointer.acceptTriple(first: PrimitivePointer, second: PrimitivePointer, third: PrimitivePointer, count: Int,
                                         action: (dst: PrimitiveType, fst: PrimitiveType, snd: PrimitiveType, trd: PrimitiveType) -> PrimitiveType) {
    require(this.isCompatibleBySize(first, count)) { "Pointers not compatible by available elements" }
    require(this.isCompatibleBySize(second, count)) { "Pointers not compatible by available elements" }
    require(this.isCompatibleBySize(third, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(first) && this.isCompatibleWith(second) && this.isCompatibleWith(third)) {
        while (end > 0) {
            val (dstBlock, dstOffset) = this.getAndIncrementBlock()
            val (fstBlock, _) = first.getAndIncrementBlock()
            val (sndBlock, _) = second.getAndIncrementBlock()
            val (trdBlock, _) = third.getAndIncrementBlock()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], fstBlock[index], sndBlock[index], trdBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(this.get(), first.getAndIncrement(), second.getAndIncrement(), third.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

@PrimitiveBinding(type1 = [DataType.BYTE, DataType.SHORT, DataType.INT, DataType.LONG,
    DataType.UBYTE, DataType.USHORT, DataType.UINT, DataType.ULONG, DataType.FLOAT, DataType.DOUBLE])
inline fun PrimitivePointer.combine(other: @Type1 PrimitivePointer, count: Int, action: (fst: PrimitiveType, snd: @Type1 PrimitiveType) -> Unit) {
    require(this.isCompatibleBySize(other, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(other)) {
        while (end > 0) {
            val (fstBlock, fstOffset) = this.getAndIncrementBlock()
            val (sndBlock, _) = other.getAndIncrementBlock()

            for (index in fstOffset until min(fstBlock.size, fstOffset + end)) {
                action(fstBlock[index], sndBlock[index])
            }

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}
