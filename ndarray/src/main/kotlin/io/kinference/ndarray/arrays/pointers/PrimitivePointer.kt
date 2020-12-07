@file:GeneratePrimitives(DataType.ALL)
@file:Suppress("DuplicatedCode")

package io.kinference.ndarray.arrays.pointers

import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlin.math.min

@GenerateNameFromPrimitives
class PrimitivePointer {
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

    @Suppress("unused")
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

    fun set(value: PrimitiveType) {
        currentBlock[indexInBlock] = value
    }

    fun get(): PrimitiveType {
        return currentBlock[indexInBlock]
    }

    fun blockIncrement() {
        when {
            blockNum < array.blocksNum - 1 -> {
                blockNum++
                indexInBlock = 0
                currentBlock = array.blocks[blockNum]
            }
            else -> indexInBlock = array.blockSize
        }
    }

    fun increment() {
        when {
            indexInBlock < array.blockSize - 1 -> indexInBlock++
            else -> blockIncrement()
        }
    }

    fun incrementAndGet(): PrimitiveType {
        increment()
        return currentBlock[indexInBlock]
    }

    fun getAndIncrement(): PrimitiveType {
        val value = currentBlock[indexInBlock]
        increment()
        return value
    }

    fun isValid(): Boolean = indexInBlock < array.blockSize
}

@BindPrimitives(type1 = [DataType.ALL])
fun PrimitivePointer.isCompatibleWith(other: @BindPrimitives.Type1 PrimitivePointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

@BindPrimitives(type1 = [DataType.ALL])
fun PrimitivePointer.isCompatibleBySize(other: @BindPrimitives.Type1 PrimitivePointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun PrimitivePointer.map(count: Int, action: (value: PrimitiveType) -> PrimitiveType) {
    var end = count
    while (end > 0) {
        val block = this.currentBlock
        val offset = this.indexInBlock
        this.blockIncrement()

        for (index in offset until min(block.size, offset + end)) {
            block[index] = action(block[index])
        }

        end -= block.size - offset
    }
}

inline fun PrimitivePointer.forEach(count: Int, action: (value: PrimitiveType) -> Unit) {
    var end = count
    while (end > 0) {
        val block = this.currentBlock
        val offset = this.indexInBlock
        this.blockIncrement()

        for (index in offset until min(block.size, offset + end)) {
            action(block[index])
        }

        end -= block.size - offset
    }
}

@BindPrimitives(type1 = [DataType.ALL])
inline fun PrimitivePointer.mapTo(container: @BindPrimitives.Type1 PrimitivePointer, count: Int, action: (value: PrimitiveType) -> @BindPrimitives.Type1 PrimitiveType) {
    require(this.isCompatibleBySize(container, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(container)) {
        while (end > 0) {
            val srcBlock = this.currentBlock
            val offset = this.indexInBlock
            this.blockIncrement()

            val dstBlock = container.currentBlock
            container.blockIncrement()

            for (index in offset until min(srcBlock.size, offset + end)) {
                dstBlock[index] = action(srcBlock[index])
            }

            end -= srcBlock.size - offset
        }
    } else {
        while (end > 0) {
            container.set(action(this.getAndIncrement()))
            container.increment()
            end--
        }
    }
}

@BindPrimitives(type1 = [DataType.ALL])
inline fun PrimitivePointer.accept(other: @BindPrimitives.Type1 PrimitivePointer, count: Int, action: (dst: PrimitiveType, src: @BindPrimitives.Type1 PrimitiveType) -> PrimitiveType) {
    require(this.isCompatibleBySize(other, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(other)) {
        while (end > 0) {
            val dstBlock = this.currentBlock
            val dstOffset = this.indexInBlock
            this.blockIncrement()

            val srcBlock = other.currentBlock
            other.blockIncrement()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], srcBlock[index])
            }

            end -= dstBlock.size - dstOffset
        }
    } else {
        while (end > 0) {
            this.set(action(this.get(), other.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

@BindPrimitives(type1 = [DataType.ALL])
inline fun PrimitivePointer.acceptWithRecursive(src: @BindPrimitives.Type1 PrimitivePointer, rec: @BindPrimitives.Type1 PrimitivePointer, count: Int,
                                                action: (dst: PrimitiveType, src: @BindPrimitives.Type1 PrimitiveType, rec: @BindPrimitives.Type1 PrimitiveType) -> PrimitiveType) {
    require(this.isCompatibleBySize(src, count)) { "Pointers not compatible by available elements" }

    var end = count
    val buf = rec.linearIndex
    if (this.isCompatibleWith(src) && this.isCompatibleWith(rec)) {
        while (end > 0) {
            if (!rec.isValid()) rec.linearIndex = buf

            val dstBlock = this.currentBlock
            val dstOffset = this.indexInBlock
            this.blockIncrement()

            val srcBlock = src.currentBlock
            src.blockIncrement()

            val recBlock = rec.currentBlock
            rec.blockIncrement()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], srcBlock[index], recBlock[index])
            }

            end -= dstBlock.size - dstOffset
        }
    } else {
        while (end > 0) {
            if (!rec.isValid()) rec.linearIndex = buf

            this.set(action(this.get(), src.getAndIncrement(), rec.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

inline fun PrimitivePointer.acceptRecursive(src: PrimitivePointer, count: Int, action: (dst: PrimitiveType, src: PrimitiveType) -> PrimitiveType) {
    var end = count
    val buf = src.linearIndex
    if (this.isCompatibleWith(src)) {
        while (end > 0) {
            if (!src.isValid()) src.linearIndex = buf

            val dstBlock = this.currentBlock
            val dstOffset = this.indexInBlock
            this.blockIncrement()

            val srcBlock = src.currentBlock
            src.blockIncrement()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], srcBlock[index])
            }

            end -= dstBlock.size - dstOffset
        }
    } else {
        while (end > 0) {
            if (!src.isValid()) src.linearIndex = buf

            this.set(action(this.get(), src.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

@BindPrimitives(type1 = [DataType.ALL])
inline fun PrimitivePointer.acceptDouble(first: @BindPrimitives.Type1 PrimitivePointer, second: @BindPrimitives.Type1 PrimitivePointer, count: Int,
                                         action: (dst: PrimitiveType, fst: @BindPrimitives.Type1 PrimitiveType, snd: @BindPrimitives.Type1 PrimitiveType) -> PrimitiveType) {
    require(this.isCompatibleBySize(first, count)) { "Pointers not compatible by available elements" }
    require(this.isCompatibleBySize(second, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(first) && this.isCompatibleWith(second)) {
        while (end > 0) {
            val dstBlock = this.currentBlock
            val dstOffset = this.indexInBlock
            this.blockIncrement()

            val fstBlock = first.currentBlock
            first.blockIncrement()

            val sndBlock = second.currentBlock
            second.blockIncrement()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size - dstOffset
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
            val dstBlock = this.currentBlock
            val dstOffset = this.indexInBlock
            this.blockIncrement()

            val fstBlock = first.currentBlock
            first.blockIncrement()

            val sndBlock = second.currentBlock
            second.blockIncrement()

            val trdBlock = third.currentBlock
            third.blockIncrement()

            for (index in dstOffset until min(dstBlock.size, dstOffset + end)) {
                dstBlock[index] = action(dstBlock[index], fstBlock[index], sndBlock[index], trdBlock[index])
            }

            end -= dstBlock.size - dstOffset
        }
    } else {
        while (end > 0) {
            this.set(action(this.get(), first.getAndIncrement(), second.getAndIncrement(), third.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

@BindPrimitives(type1 = [DataType.ALL])
inline fun PrimitivePointer.combine(other: @BindPrimitives.Type1 PrimitivePointer, count: Int,
                                    action: (fst: PrimitiveType, snd: @BindPrimitives.Type1 PrimitiveType) -> Unit) {
    require(this.isCompatibleBySize(other, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(other)) {
        while (end > 0) {
            val fstBlock = this.currentBlock
            val fstOffset = this.indexInBlock
            this.blockIncrement()

            val sndBlock = other.currentBlock
            other.blockIncrement()

            for (index in fstOffset until min(fstBlock.size, fstOffset + end)) {
                action(fstBlock[index], sndBlock[index])
            }

            end -= fstBlock.size - fstOffset
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}
