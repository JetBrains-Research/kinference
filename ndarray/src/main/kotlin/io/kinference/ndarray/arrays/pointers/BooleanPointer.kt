@file:Suppress("NOTHING_TO_INLINE", "EXPERIMENTAL_API_USAGE", "DuplicatedCode")

package io.kinference.ndarray.arrays.pointers


import io.kinference.ndarray.arrays.tiled.BooleanTiledArray
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import kotlin.math.min

class BooleanPointer {
    data class BlockWithOffset(val block: BooleanArray, val offset: Int)

    val array: BooleanTiledArray

    var blockNum: Int
    var indexInBlock: Int

    var currentBlock: BooleanArray

    constructor(array: BooleanTiledArray, startIndex: Int = 0) {
        require(startIndex >= 0 && startIndex < array.size) { "Start index of Iterator must be >= 0 and < array size" }
        this.array = array
        this.blockNum = startIndex / array.blockSize
        this.indexInBlock = startIndex % array.blockSize
        this.currentBlock = array.blocks[blockNum]
    }

    constructor(other: BooleanPointer) {
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

    inline fun set(value: Boolean) {
        currentBlock[indexInBlock] = value
    }

    inline fun get(): Boolean {
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

    inline fun incrementAndGet(): Boolean {
        increment()
        return currentBlock[indexInBlock]
    }

    inline fun getAndIncrement(): Boolean {
        val value = currentBlock[indexInBlock]
        increment()
        return value
    }

    fun isValid(): Boolean = indexInBlock < array.blockSize
}


inline fun BooleanPointer.isCompatibleWith(other: BytePointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: ShortPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: IntPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: LongPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: BooleanPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: UBytePointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: UShortPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: UIntPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: ULongPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: FloatPointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}

inline fun BooleanPointer.isCompatibleWith(other: DoublePointer): Boolean {
    return this.indexInBlock == other.indexInBlock && this.array.blockSize == other.array.blockSize
}



inline fun BooleanPointer.isCompatibleBySize(other: BytePointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: ShortPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: IntPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: LongPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: BooleanPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: UBytePointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: UShortPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: UIntPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: ULongPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: FloatPointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}

inline fun BooleanPointer.isCompatibleBySize(other: DoublePointer, requestedSize: Int): Boolean {
    return this.array.size - this.linearIndex >= requestedSize && other.array.size - other.linearIndex >= requestedSize
}


fun BooleanPointer.map(count: Int, action: (value: Boolean) -> Boolean) {
    var end = count
    while (end > 0) {
        val block = this.currentBlock
        val offset = this.indexInBlock
        this.blockIncrement()

        for (index in offset until min(block.size, offset + end)) {
            block[index] = action(block[index])
        }

        end -= block.size
    }
}

fun BooleanPointer.forEach(count: Int, action: (value: Boolean) -> Unit) {
    var end = count
    while (end > 0) {
        val block = this.currentBlock
        val offset = this.indexInBlock
        this.blockIncrement()

        for (index in offset until min(block.size, offset + end)) {
            action(block[index])
        }

        end -= block.size
    }
}


fun BooleanPointer.mapTo(container: BytePointer, count: Int, action: (value: Boolean) -> Byte) {
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

fun BooleanPointer.mapTo(container: ShortPointer, count: Int, action: (value: Boolean) -> Short) {
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

fun BooleanPointer.mapTo(container: IntPointer, count: Int, action: (value: Boolean) -> Int) {
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

fun BooleanPointer.mapTo(container: LongPointer, count: Int, action: (value: Boolean) -> Long) {
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

fun BooleanPointer.mapTo(container: UBytePointer, count: Int, action: (value: Boolean) -> UByte) {
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

fun BooleanPointer.mapTo(container: UShortPointer, count: Int, action: (value: Boolean) -> UShort) {
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

fun BooleanPointer.mapTo(container: UIntPointer, count: Int, action: (value: Boolean) -> UInt) {
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

fun BooleanPointer.mapTo(container: ULongPointer, count: Int, action: (value: Boolean) -> ULong) {
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

fun BooleanPointer.mapTo(container: FloatPointer, count: Int, action: (value: Boolean) -> Float) {
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

fun BooleanPointer.mapTo(container: DoublePointer, count: Int, action: (value: Boolean) -> Double) {
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



fun BooleanPointer.accept(other: BytePointer, count: Int, action: (dst: Boolean, src: Byte) -> Boolean) {
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

fun BooleanPointer.accept(other: ShortPointer, count: Int, action: (dst: Boolean, src: Short) -> Boolean) {
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

fun BooleanPointer.accept(other: IntPointer, count: Int, action: (dst: Boolean, src: Int) -> Boolean) {
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

fun BooleanPointer.accept(other: LongPointer, count: Int, action: (dst: Boolean, src: Long) -> Boolean) {
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

fun BooleanPointer.accept(other: UBytePointer, count: Int, action: (dst: Boolean, src: UByte) -> Boolean) {
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

fun BooleanPointer.accept(other: UShortPointer, count: Int, action: (dst: Boolean, src: UShort) -> Boolean) {
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

fun BooleanPointer.accept(other: UIntPointer, count: Int, action: (dst: Boolean, src: UInt) -> Boolean) {
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

fun BooleanPointer.accept(other: ULongPointer, count: Int, action: (dst: Boolean, src: ULong) -> Boolean) {
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

fun BooleanPointer.accept(other: FloatPointer, count: Int, action: (dst: Boolean, src: Float) -> Boolean) {
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

fun BooleanPointer.accept(other: DoublePointer, count: Int, action: (dst: Boolean, src: Double) -> Boolean) {
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

fun BooleanPointer.accept(other: BooleanPointer, count: Int, action: (dst: Boolean, src: Boolean) -> Boolean) {
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


fun BooleanPointer.acceptDouble(first: BooleanPointer, second: BooleanPointer, count: Int, action: (dst: Boolean, fst: Boolean, snd: Boolean) -> Boolean) {
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

fun BooleanPointer.acceptDouble(first: BytePointer, second: BytePointer, count: Int, action: (fst: Byte, snd: Byte) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: ShortPointer, second: ShortPointer, count: Int, action: (fst: Short, snd: Short) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: IntPointer, second: IntPointer, count: Int, action: (fst: Int, snd: Int) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: LongPointer, second: LongPointer, count: Int, action: (fst: Long, snd: Long) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: UBytePointer, second: UBytePointer, count: Int, action: (fst: UByte, snd: UByte) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: UShortPointer, second: UShortPointer, count: Int, action: (fst: UShort, snd: UShort) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: UIntPointer, second: UIntPointer, count: Int, action: (fst: UInt, snd: UInt) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: ULongPointer, second: ULongPointer, count: Int, action: (fst: ULong, snd: ULong) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: FloatPointer, second: FloatPointer, count: Int, action: (fst: Float, snd: Float) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptDouble(first: DoublePointer, second: DoublePointer, count: Int, action: (fst: Double, snd: Double) -> Boolean) {
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
                dstBlock[index] = action(fstBlock[index], sndBlock[index])
            }

            end -= dstBlock.size
        }
    } else {
        while (end > 0) {
            this.set(action(first.getAndIncrement(), second.getAndIncrement()))
            this.increment()
            end--
        }
    }
}

fun BooleanPointer.acceptTriple(first: BooleanPointer, second: BooleanPointer, third: BooleanPointer, count: Int,
                                    action: (dst: Boolean, fst: Boolean, snd: Boolean, trd: Boolean) -> Boolean) {
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


fun BooleanPointer.combine(other: BytePointer, count: Int, action: (fst: Boolean, snd: Byte) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: ShortPointer, count: Int, action: (fst: Boolean, snd: Short) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: IntPointer, count: Int, action: (fst: Boolean, snd: Int) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: LongPointer, count: Int, action: (fst: Boolean, snd: Long) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: UBytePointer, count: Int, action: (fst: Boolean, snd: UByte) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: UShortPointer, count: Int, action: (fst: Boolean, snd: UShort) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: UIntPointer, count: Int, action: (fst: Boolean, snd: UInt) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: ULongPointer, count: Int, action: (fst: Boolean, snd: ULong) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: FloatPointer, count: Int, action: (fst: Boolean, snd: Float) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

fun BooleanPointer.combine(other: DoublePointer, count: Int, action: (fst: Boolean, snd: Double) -> Unit) {
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

            end -= fstBlock.size
        }
    } else {
        while (end > 0) {
            action(this.getAndIncrement(), other.getAndIncrement())
            end--
        }
    }
}

