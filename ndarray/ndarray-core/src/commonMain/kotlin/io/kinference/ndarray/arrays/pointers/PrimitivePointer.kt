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

    fun blockDecrement() {
        when {
            blockNum > 0 -> {
                blockNum--
                indexInBlock = array.blockSize - 1
                currentBlock = array.blocks[blockNum]
            }
            else -> indexInBlock = -1
        }
    }

    fun decrement() {
        when {
            indexInBlock > 0 -> indexInBlock--
            else -> blockDecrement()
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

    fun setAndIncrement(value: PrimitiveType) {
        currentBlock[indexInBlock] = value
        increment()
    }

    fun isValid(): Boolean = indexInBlock < array.blockSize && indexInBlock > -1
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

        if (block.size <= offset + end) {
            this.blockIncrement()
        } else {
            this.indexInBlock += offset + end
        }

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

        if (block.size <= offset + end) {
            this.blockIncrement()
        } else {
            this.indexInBlock += end
        }

        for (index in offset until min(block.size, offset + end)) {
            action(block[index])
        }

        end -= block.size - offset
    }
}

inline fun PrimitivePointer.forEachIndexed(count: Int, startIndex: Int = 0, action: (index: Int, value: PrimitiveType) -> Unit) {
    var end = count
    var idx = startIndex
    while (end > 0) {
        val block = this.currentBlock
        val offset = this.indexInBlock

        if (block.size <= offset + end) {
            this.blockIncrement()
        } else {
            this.indexInBlock += end
        }

        for (index in offset until min(block.size, offset + end)) {
            action(idx++, block[index])
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

            val dstBlock = container.currentBlock

            if (srcBlock.size <= offset + end) {
                this.blockIncrement()
                container.blockIncrement()
            } else {
                this.indexInBlock += end
                container.indexInBlock += end
            }

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

            val srcBlock = other.currentBlock

            if (dstBlock.size <= dstOffset + end) {
                this.blockIncrement()
                other.blockIncrement()
            } else {
                this.indexInBlock += end
                other.indexInBlock += end
            }

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

            val srcBlock = src.currentBlock

            val recBlock = rec.currentBlock

            if (dstBlock.size <= dstOffset + end) {
                this.blockIncrement()
                src.blockIncrement()
                rec.blockIncrement()
            } else {
                this.indexInBlock += end
                src.indexInBlock += end
                rec.indexInBlock += end
            }

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

            val srcBlock = src.currentBlock

            if (dstBlock.size <= dstOffset + end) {
                this.blockIncrement()
                src.blockIncrement()
            } else {
                this.indexInBlock += end
                src.indexInBlock += end
            }

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

            val fstBlock = first.currentBlock

            val sndBlock = second.currentBlock

            if (dstBlock.size <= dstOffset + end) {
                this.blockIncrement()
                first.blockIncrement()
                second.blockIncrement()
            } else {
                this.indexInBlock += end
                first.indexInBlock += end
                second.indexInBlock += end
            }

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

@BindPrimitives(type1 = [DataType.ALL])
inline fun PrimitivePointer.acceptTriple(
    first: PrimitivePointer, second: PrimitivePointer, third: @BindPrimitives.Type1 PrimitivePointer, count: Int,
    action: (dst: PrimitiveType, fst: PrimitiveType, snd: PrimitiveType, trd: @BindPrimitives.Type1 PrimitiveType) -> PrimitiveType
) {
    require(this.isCompatibleBySize(first, count)) { "Pointers not compatible by available elements" }
    require(this.isCompatibleBySize(second, count)) { "Pointers not compatible by available elements" }
    require(this.isCompatibleBySize(third, count)) { "Pointers not compatible by available elements" }

    var end = count
    if (this.isCompatibleWith(first) && this.isCompatibleWith(second) && this.isCompatibleWith(third)) {
        while (end > 0) {
            val dstBlock = this.currentBlock
            val dstOffset = this.indexInBlock

            val fstBlock = first.currentBlock

            val sndBlock = second.currentBlock

            val trdBlock = third.currentBlock

            if (dstBlock.size <= dstOffset + end) {
                this.blockIncrement()
                first.blockIncrement()
                second.blockIncrement()
                third.blockIncrement()
            } else {
                this.indexInBlock += end
                first.indexInBlock += end
                second.indexInBlock += end
                third.indexInBlock += end
            }

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

            val sndBlock = other.currentBlock

            if (fstBlock.size <= fstOffset + end) {
                this.blockIncrement()
                other.blockIncrement()
            } else {
                this.indexInBlock += end
                other.indexInBlock += end
            }

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
