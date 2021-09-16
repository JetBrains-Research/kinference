@file:GeneratePrimitives(DataType.NUMBER)
@file:Suppress("DuplicatedCode", "unused")

package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlinx.coroutines.*
import kotlin.math.*

@GenerateNameFromPrimitives
open class PrimitiveNDArray(array: PrimitiveTiledArray, strides: Strides) : NumberNDArray {
    constructor(shape: IntArray) : this(PrimitiveTiledArray(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> PrimitiveType) : this(PrimitiveTiledArray(shape, init), Strides(shape))

    constructor(strides: Strides) : this(PrimitiveTiledArray(strides), strides)
    constructor(strides: Strides, init: (Int) -> PrimitiveType) : this(PrimitiveTiledArray(strides, init), strides)

    var array: PrimitiveTiledArray = array
        protected set

    internal val blocksInRow: Int
        get() = when {
            strides.linearSize == 0 -> 0
            strides.shape.isEmpty() -> 1
            else -> strides.shape.last() / array.blockSize
        }

    override fun view(vararg axes: Int): PrimitiveNDArray {
        for ((i, axis) in axes.withIndex()) {
            require(shape[i] > axis)
        }

        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        if (array.blockSize == 0)
            return PrimitiveNDArray(array, newStrides)


        val offsetBlocks = offset / array.blockSize

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = PrimitiveTiledArray(copyBlocks)

        return PrimitiveNDArray(newArray, newStrides)
    }

    override val type = DataType.CurrentPrimitive

    final override var strides: Strides = strides
        protected set


    override fun singleValue(): PrimitiveType {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array.blocks[0][0]
    }

    override fun allocateNDArray(strides: Strides): MutablePrimitiveNDArray = MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides)

    override fun reshapeView(newShape: IntArray): NDArray {
        val newStrides = Strides(newShape)

        require(newStrides.linearSize == linearSize)

        return PrimitiveNDArray(array, newStrides)
    }

    override fun toMutable(newStrides: Strides): MutableNumberNDArray = MutablePrimitiveNDArray(array.copyOf(), newStrides)

    override fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNumberNDArray {
        function as PrimitiveMap
        destination as MutablePrimitiveNDArray

        for (blockNum in 0 until array.blocksNum) {
            val thisBlock = this.array.blocks[blockNum]
            val destBlock = destination.array.blocks[blockNum]

            for (idx in thisBlock.indices) {
                destBlock[idx] = function.apply(thisBlock[idx])
            }
        }
        return destination
    }

    override fun erfFor(value: Any): PrimitiveType {
        value as PrimitiveType
        val sign = value.toDouble().sign
        val doubleValue = abs(value.toDouble())
        val t = 1 / (1 + ERF_P_VALUE * doubleValue)

        val sum = t * (ERF_COEF[0] + t * (ERF_COEF[1] + t * (ERF_COEF[2] + t * (ERF_COEF[3] + t * ERF_COEF[4]))))

        return (sign * (1.0 - sum * exp(-doubleValue * doubleValue))).toPrimitive()
    }

    override fun withZeroPoint(zeroPoint: NumberNDArray): IntNDArray {
        zeroPoint as PrimitiveNDArray

        return if (zeroPoint.linearSize == 1) {
            val zero = zeroPoint.array.blocks[0][0].toInt()
            val arr = IntTiledArray(this.strides)
            arr.pointer().accept(array.pointer(), arr.size) { _, src -> src.toInt() - zero }
            IntNDArray(arr, strides)
        } else {
            val arr = IntTiledArray(strides)
            arr.pointer().acceptWithRecursive(this.array.pointer(), zeroPoint.array.pointer(), arr.size) { _, src, zero -> src.toInt() - zero.toInt() }
            IntNDArray(arr, strides)
        }
    }

    @Suppress("CAST_NEVER_SUCCEEDS")
    override fun dequantize(zeroPoint: NDArray?, scale: NDArray, axis: Int?): NDArray {
        scale as FloatNDArray
        val zeros = (zeroPoint as? PrimitiveNDArray)?.array
        val output = MutableFloatNDArray(FloatTiledArray(this.array.size, this.array.blockSize), this.strides)

        when {
            canDequantizePerTensor(zeroPoint, scale) -> {
                val zero = if (zeros == null) 0f else zeros.blocks[0][0].toFloat()
                val sc = scale.array.blocks[0][0]

                if (type == DataType.BYTE) {
                    output.array.pointer().accept(this.array.pointer() as BytePointer, output.linearSize) { _, src ->
                        (src.toFloat() - zero) * sc
                    }
                } else {
                    output.array.pointer().accept(this.array.pointer() as UBytePointer, output.linearSize) { _, src ->
                        (src.toFloat() - zero) * sc
                    }
                }
            }
            canDequantizePerAxis(axis!!, zeroPoint, scale) -> {
                val actualAxis = indexAxis(axis)
                val blockCount = computeBlockSize(toDim = actualAxis)
                val blockSize = computeBlockSize(fromDim = actualAxis + 1)
                var outOffset = 0
                repeat(blockCount) {
                    val zeroPointer = zeros?.pointer()
                    val scalePointer = scale.array.pointer()
                    for (i in 0 until shape[actualAxis]) {
                        val zero = zeroPointer?.getAndIncrement()?.toFloat() ?: 0f
                        val sc = scalePointer.getAndIncrement()

                        if (type == DataType.BYTE) {
                            output.array.pointer(outOffset).accept(this.array.pointer(outOffset) as BytePointer, blockSize) { _, src ->
                                (src.toFloat() - zero) * sc
                            }
                        } else {
                            output.array.pointer(outOffset).accept(this.array.pointer(outOffset) as UBytePointer, blockSize) { _, src ->
                                (src.toFloat() - zero) * sc
                            }
                        }

                        outOffset += blockSize
                    }
                }
            }
            else -> error("Cannot perform dequantization. Scale and zero point tensors should be either scalars or 1D tensors containing ${shape[axis]} elements")
        }

        return output
    }

    override fun row(row: Int): MutableNumberNDArray {
        val rowLength: Int = linearSize / shape[0]
        val start = row * rowLength
        val dims = shape.copyOfRange(1, rank)

        val result = PrimitiveTiledArray(Strides(dims))
        result.pointer().accept(array.pointer(start), result.size) { _, src -> src }

        return MutablePrimitiveNDArray(result, Strides(dims))
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArray {
        val newShape = IntArray(shape.size) {
            val length = abs(ends[it] - starts[it])
            val rest = length % abs(steps[it])
            (length / abs(steps[it])) + if (rest != 0) 1 else 0
        }

        val newStrides = Strides(newShape)
        val newArray = PrimitiveTiledArray(newStrides)

        if (newArray.size > 0) {
            slice(newArray.pointer(), this.array.pointer(), 0, 0, shape, starts, ends, steps)
        }

        return MutablePrimitiveNDArray(newArray, newStrides)
    }

    private fun slice(dst: PrimitivePointer, src: PrimitivePointer, offset: Int, axis: Int, shape: IntArray, starts: IntArray, ends: IntArray, steps: IntArray) {
        val start = starts[axis]
        val end = ends[axis]
        val step = steps[axis]

        val range = if (step > 0) (start until end step step) else (start downTo end + 1 step -step)

        if (axis == shape.size - 1) {
            for (index in range) {
                src.linearIndex = offset + index
                dst.set(src.get())
                dst.increment()

                /*
                        for (index in range) {
            array.putNext(this.array[additionalOffset + index])
        }
                 */
            }
        } else {
            var dim = 1
            for (ind in (axis + 1) until shape.size) dim *= shape[ind]

            for (index in range) {
                slice(dst, src, offset + index * dim, axis + 1, shape, starts, ends, steps)
            }
        }
    }

    override fun min(): PrimitiveType {
        var min = PrimitiveType.MAX_VALUE
        for (block in array.blocks) {
            for (idx in block.indices) {
                val tmp = block[idx]
                if (tmp < min) min = tmp
            }
        }
        return min
    }

    override fun max(): PrimitiveType {
        var max = PrimitiveType.MIN_VALUE
        for (block in array.blocks) {
            for (idx in block.indices) {
                val tmp = block[idx]
                if (tmp > max) max = tmp
            }
        }

        return max
    }

    override fun sum(): PrimitiveType {
        var sum = (0).toPrimitive()

        for (block in array.blocks) {
            for (idx in block.indices) {
                sum = (sum + block[idx]).toPrimitive()
            }
        }
        return sum
    }

    override fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArray {
        val output = MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides)

        val actualAxis = indexAxis(axis)

        val blockSize = computeBlockSize(fromDim = actualAxis + 1)
        val batchSize = computeBlockSize(fromDim = actualAxis)
        val numBatches = computeBlockSize(toDim = actualAxis)
        val numBlocks = batchSize / blockSize
        repeat(numBatches) { batchIdx ->
            val dstOff = if (!reverse) batchIdx * batchSize else (numBatches - batchIdx) * batchSize - 1
            if (!exclusive) {
                if (!reverse)
                    output.copyFrom(dstOff, this, dstOff, dstOff + blockSize)
                else
                    output.copyFrom(dstOff - blockSize + 1, this, dstOff - blockSize + 1, dstOff + 1)
            }

            // TODO rewrite using pointers
            if (!reverse) {
                for (i in 1 until numBlocks) {
                    for (j in 0 until blockSize) {
                        val currentOff = dstOff + i * blockSize + j
                        val thisOff = if (!exclusive) currentOff else currentOff - blockSize
                        output.array[currentOff] = (output.array[currentOff - blockSize] + array[thisOff]).toPrimitive()
                    }
                }
            } else {
                for (i in 1 until numBlocks) {
                    for (j in blockSize - 1 downTo 0) {
                        val currentOff = dstOff - i * blockSize - j
                        val thisOff = if (!exclusive) currentOff else currentOff + blockSize
                        output.array[currentOff] = (output.array[currentOff + blockSize] + array[thisOff]).toPrimitive()
                    }
                }
            }
        }

        return output
    }

    override fun plus(other: NumberNDArray): MutableNumberNDArray {
        val destShape = Broadcasting.broadcastShape(listOf(this.shape, other.shape))
        val destStrides = Strides(destShape)
        return plus(other, MutablePrimitiveNDArray(PrimitiveTiledArray(destStrides), destStrides))
    }

    private fun plusScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (arrayBlock[idx] + scalar).toPrimitive()
            }
        }
    }

    override fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (array.blocks[0][0] + other.array.blocks[0][0]).toPrimitive()
            this.isScalar() -> plusScalar(other.array, this.array.blocks[0][0], destination.array)
            other.isScalar() -> plusScalar(this.array, other.array.blocks[0][0], destination.array)
            else -> this.applyWithBroadcast(other, destination, false) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray
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

        return destination
    }

    override fun minus(other: NumberNDArray): MutableNumberNDArray {
        val destShape = Broadcasting.broadcastShape(listOf(this.shape, other.shape))
        val destStrides = Strides(destShape)
        return minus(other, MutablePrimitiveNDArray(PrimitiveTiledArray(destStrides), destStrides))
    }

    private fun minusScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (arrayBlock[idx] - scalar).toPrimitive()
            }
        }
    }

    private fun minusFromScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (scalar - arrayBlock[idx]).toPrimitive()
            }
        }
    }

    override fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (this.array.blocks[0][0] - other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> minusScalar(array, other.array.blocks[0][0], destination.array)
            this.isScalar() -> minusFromScalar(other.array, array.blocks[0][0], destination.array)
            else -> this.applyWithBroadcast(other, destination, true) { left, right, dest ->
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

        return destination
    }

    override fun times(other: NumberNDArray): MutableNumberNDArray {
        val destShape = Broadcasting.broadcastShape(listOf(this.shape, other.shape))
        val destStrides = Strides(destShape)
        return times(other, MutablePrimitiveNDArray(PrimitiveTiledArray(destStrides), destStrides))
    }

    private fun timesScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (arrayBlock[idx] * scalar).toPrimitive()
            }
        }
    }

    override fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (this.array.blocks[0][0] * other.array.blocks[0][0]).toPrimitive()
            this.isScalar() -> timesScalar(other.array, this.array.blocks[0][0], destination.array)
            other.isScalar() -> timesScalar(this.array, other.array.blocks[0][0], destination.array)
            else -> this.applyWithBroadcast(other, destination, false) { left, right, dest ->
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

        return destination
    }

    override fun div(other: NumberNDArray): MutableNumberNDArray {
        val destShape = Broadcasting.broadcastShape(listOf(this.shape, other.shape))
        val destStrides = Strides(destShape)
        return div(other, MutablePrimitiveNDArray(PrimitiveTiledArray(destStrides), destStrides))
    }

    private fun divByScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (arrayBlock[idx] / scalar).toPrimitive()
            }
        }
    }

    private fun divScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (scalar / arrayBlock[idx]).toPrimitive()
            }
        }
    }

    override fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (this.array.blocks[0][0] / other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> divByScalar(array, other.array.blocks[0][0], destination.array)
            this.isScalar() -> divScalar(other.array, array.blocks[0][0], destination.array)
            else -> this.applyWithBroadcast(other, destination, true) { left, right, dest ->
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

        return destination
    }

    private fun resortBlocks(blocks: Array<PrimitiveArray>, colSize: Int, blocksInRow: Int): Array<PrimitiveArray> {
        if (blocks.size == 1) return blocks
        val result = blocks.copyOf()

        for (i in 0 until blocksInRow) {
            for (j in 0 until colSize) {
                result[i * colSize + j] = blocks[j * blocksInRow + i]
            }
        }

        return result
    }

    override fun dot(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        other as PrimitiveNDArray; destination as MutablePrimitiveNDArray
        require(shape.size in 1..2 && other.shape.size in 1..2)
        val actualThis = (if (this.shape.size == 1) this.reshapeView(intArrayOf(1, shape[0])) else this) as PrimitiveNDArray
        val actualOther = (if (other.shape.size == 1) other.reshapeView(intArrayOf(1, other.shape[0])) else other) as PrimitiveNDArray

        require(actualThis.shape[1] == actualOther.shape[0])

        val n = actualThis.shape[0]
        val t = actualThis.shape[1]

        val resortedLeft = resortBlocks(actualThis.array.blocks, actualThis.shape[0], actualThis.blocksInRow)
        val resortedRight = resortBlocks(actualOther.array.blocks, actualOther.shape[0], actualOther.blocksInRow)
        val resortedDest = resortBlocks(destination.array.blocks, destination.shape[0], destination.blocksInRow)

        val lBlockSize = actualThis.array.blockSize
        val rdBlockSize = destination.array.blockSize

        val lBlockInRow = actualThis.blocksInRow
        val rBlockInRow = other.blocksInRow

        fun wrapper(body: (inner: () -> Unit) -> Unit = { it() }) {
            for (rdCol in 0 until rBlockInRow) {
                val rightIdx = rdCol * t
                val destIdx = rdCol * n

                body {
                    for (i in 0 until n) {
                        val destBlock = resortedDest[destIdx + i]
                        for (lCol in 0 until lBlockInRow) {
                            val leftBlock = resortedLeft[i + lCol * n]
                            val rightIdxOffset = rightIdx + lBlockSize * lCol

                            for (k in 0 until lBlockSize) {
                                val temp = leftBlock[k]
                                val rightBlock = resortedRight[rightIdxOffset + k]

                                for (j in 0 until rdBlockSize) {
                                    destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toPrimitive()
                                }
                            }
                        }
                    }
                }
            }
        }

        if (rBlockInRow > 1) {
            runBlocking(Dispatchers.Default) { wrapper { launch { it() } } }
        } else {
            wrapper()
        }

        return destination
    }

    override fun dotTransposedWithAlpha(alpha: Double, other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        other as PrimitiveNDArray; destination as MutablePrimitiveNDArray

        @Suppress("NAME_SHADOWING") val alpha = alpha.toPrimitive()
        val dRowsNum = destination.shape[0]

        val dBlocksInRow = destination.blocksInRow
        val lrBlocksInRow = this.blocksInRow

        val dBlockSize = destination.array.blockSize
        val lrBlockSize = array.blockSize

        val dBlocks = destination.array.blocks
        val lBlocks = this.array.blocks
        val rBlocks = other.array.blocks

        fun wrapper(body: (inner: () -> Unit) -> Unit = { it() }) {
            for (dRow in 0 until dRowsNum) {
                var rRow = 0
                val dBlockOffset = dRow * dBlocksInRow
                val lBlockOffset = dRow * lrBlocksInRow

                for (dBlockInRow in 0 until dBlocksInRow) {
                    val dBlock = dBlocks[dBlockOffset + dBlockInRow]
                    val rRowOffset = rRow
                    body {
                        var rBlockOffset = rRowOffset * lrBlocksInRow
                        for (dIdx in 0 until dBlockSize) {
                            for (lrBlockInRow in 0 until lrBlocksInRow) {
                                val lBlock = lBlocks[lBlockOffset + lrBlockInRow]
                                val rBlock = rBlocks[rBlockOffset + lrBlockInRow]

                                for (lrIdx in 0 until lrBlockSize) {
                                    dBlock[dIdx] = (dBlock[dIdx] + alpha * lBlock[lrIdx] * rBlock[lrIdx]).toPrimitive()
                                }
                            }
                            rBlockOffset += lrBlocksInRow
                        }
                    }

                    rRow += dBlockSize
                }
            }
        }

        if (destination.blocksInRow > 1) {
            runBlocking(Dispatchers.Default) { wrapper { launch { it() } } }
        } else {
            wrapper()
        }

        return destination
    }

    override fun gemm(m: Int, n: Int, k: Int, alpha: Double, lda: Int, b: NDArray, ldb: Int, beta: Double, c: MutableNDArray, ldc: Int, aOffset: Int, bOffset: Int, cOffset: Int, transposeA: Boolean, transposeB: Boolean): MutableNDArray {
        b as PrimitiveNDArray; c as MutablePrimitiveNDArray
        val betaPrimitive = beta.toPrimitive()
        val alphaPrimitive = alpha.toPrimitive()
        val aPointer = array.pointer()
        val bPointer = b.array.pointer()
        val cPointer = c.array.pointer()

        if (beta != 1.0) {
            for (i in 0 until m) {
                cPointer.linearIndex = i * ldc + cOffset
                cPointer.map(n) { (betaPrimitive * it).toPrimitive() }
            }
        }

        when {
            transposeA && transposeB -> {
                // TODO rewrite using block operations
                for (t in 0 until m) {
                    for (j in 0 until n) {
                        cPointer.linearIndex = t * ldc + j + cOffset
                        for (i in 0 until k) {
                            aPointer.linearIndex = i * lda + t + aOffset
                            bPointer.linearIndex = j * ldb + i + bOffset
                            cPointer.set((alphaPrimitive * aPointer.get() * bPointer.get() + cPointer.get()).toPrimitive())
                        }
                    }
                }
            }
            transposeA -> {
                // TODO rewrite using block operations
                for (t in 0 until m) {
                    for (j in 0 until n) {
                        cPointer.linearIndex = t * ldc + j + cOffset
                        for (i in 0 until k) {
                            aPointer.linearIndex = i * lda + t + aOffset
                            bPointer.linearIndex = i * ldb + j + bOffset
                            cPointer.set((alphaPrimitive * aPointer.get() * bPointer.get() + cPointer.get()).toPrimitive())
                        }
                    }
                }
            }
            transposeB -> {
                for (t in 0 until m) {
                    val aIdx = t * lda + aOffset
                    cPointer.linearIndex = t * ldc + cOffset
                    for (j in 0 until n) {
                        aPointer.linearIndex = aIdx
                        bPointer.linearIndex = j * ldb + bOffset

                        aPointer.combine(bPointer, k) { elementInA, elementInB ->
                            cPointer.set((alphaPrimitive * elementInA * elementInB + cPointer.get()).toPrimitive())
                        }

                        cPointer.increment()
                    }
                }
            }
            else -> {
                for (t in 0 until m) {
                    val cIdx = t * ldc + cOffset
                    aPointer.linearIndex = t * lda + aOffset
                    for (i in 0 until k) {
                        val temp = (alphaPrimitive * aPointer.getAndIncrement()).toPrimitive()

                        bPointer.linearIndex = i * ldb + bOffset
                        cPointer.linearIndex = cIdx

                        cPointer.accept(bPointer, n) { elementInC, elementInB ->
                            (temp * elementInB + elementInC).toPrimitive()
                        }
                    }
                }
            }
        }

        return c
    }

    override fun argmax(axis: Int, keepDims: Boolean, selectLastIndex: Boolean): IntNDArray {
        val actualAxis = indexAxis(axis)

        val countIterations = shape.sliceArray(0 until actualAxis).fold(1) { acc, i -> acc * i }
        val countElements = shape.sliceArray((actualAxis + 1) until rank).fold(1) { acc, i -> acc * i }
        val countDims = shape[actualAxis]

        val outputShape = if (keepDims) shape.copyOf().apply { set(actualAxis, 1) } else shape.sliceArray(shape.indices.minus(actualAxis))
        val outputArray = allocateNDArray(DataType.INT, outputShape) as MutableIntNDArray
        val tempMaxValues = if (actualAxis == shape.lastIndex) PrimitiveTiledArray(1, 1) else PrimitiveTiledArray(countElements, outputArray.array.blockSize)

        val inputPointer = this.array.pointer()

        for (i in 0 until countIterations) {
            var maxValuesPointer = tempMaxValues.pointer()

            maxValuesPointer.accept(inputPointer, countElements) { _, src -> src }

            for (j in 1 until countDims) {
                val outputPointer = outputArray.array.pointer(i * countElements)
                maxValuesPointer = tempMaxValues.pointer()

                var end = countElements

                if (inputPointer.isCompatibleWith(outputPointer) && inputPointer.isCompatibleWith(maxValuesPointer)) {
                    while (end > 0) {
                        val outputBlock = outputPointer.currentBlock
                        val offset = outputPointer.indexInBlock
                        outputPointer.blockIncrement()

                        val inputBlock = inputPointer.currentBlock
                        inputPointer.blockIncrement()

                        val maxValuesBlock = maxValuesPointer.currentBlock
                        maxValuesPointer.blockIncrement()

                        for (index in offset until min(outputBlock.size, offset + end)) {
                            val value = inputBlock[index]
                            val oldMaxValue = maxValuesBlock[index]
                            if (value > oldMaxValue) {
                                maxValuesBlock[index] = value
                                outputBlock[index] = j
                            } else if (selectLastIndex && value == oldMaxValue) {
                                outputBlock[index] = j
                            }
                        }

                        end -= outputBlock.size - offset
                    }
                } else {
                    while (end > 0) {
                        val value = inputPointer.getAndIncrement()
                        val oldMaxValue = maxValuesPointer.get()

                        if (value > oldMaxValue) {
                            maxValuesPointer.set(value)
                            outputPointer.set(j)
                        } else if (selectLastIndex && value == oldMaxValue) {
                            outputPointer.set(j)
                        }

                        maxValuesPointer.increment()
                        outputPointer.increment()
                        end--
                    }
                }
            }
        }

        return outputArray
    }

    override fun reduceSum(axes: IntArray, keepDims: Boolean): PrimitiveNDArray {
        val actualAxes = axes.map { indexAxis(it) }.toSet().sorted()
        require(actualAxes.all { it in shape.indices }) { "Axes ${axes.joinToString()} must be in range [-$rank, ${rank - 1}]" }

        return actualAxes.foldIndexed(this) { index: Int, acc: PrimitiveNDArray, axis: Int ->
            if (keepDims) {
                acc.reduceSum(axis, keepDims)
            } else {
                acc.reduceSum(axis - index, keepDims)
            }
        }
    }

    override fun reduceSum(axis: Int, keepDims: Boolean): PrimitiveNDArray {
        val actualAxis = indexAxis(axis)

        val outputShape = if (keepDims) shape.copyOf().apply { set(actualAxis, 1) } else shape.sliceArray(shape.indices.minus(actualAxis))
        val outputArray = allocateNDArray(Strides(outputShape))

        val countIterations = shape.sliceArray(0 until axis).fold(1) { acc, i -> acc * i }
        val countElements = shape.sliceArray((axis + 1) until rank).fold(1) { acc, i -> acc * i }
        val countDims = shape[axis]

        val inputPointer = this.array.pointer()
        val outputPointer = outputArray.array.pointer()

        if (axis == shape.lastIndex) {
            repeat(countIterations) {
                var sumAlongLastDim = (0).toPrimitive()
                inputPointer.forEach(countDims) { sumAlongLastDim = (sumAlongLastDim + it).toPrimitive() }
                outputPointer.set(sumAlongLastDim)
                outputPointer.increment()
            }
        } else {
            repeat(countIterations) { iteration ->
                repeat(countDims) {
                    outputPointer.linearIndex = iteration * countElements
                    outputPointer.accept(inputPointer, countElements) { dst: PrimitiveType, src: PrimitiveType ->
                        (dst + src).toPrimitive()
                    }
                }
            }
        }

        return outputArray
    }

    override fun topK(axis: Int, k: Int, largest: Boolean, sorted: Boolean): Pair<PrimitiveNDArray, LongNDArray> {

        val actualAxis = indexAxis(axis)

        val outputStrides = Strides(shape.copyOf().apply { set(actualAxis, k) })


        val outputArray = allocateNDArray(outputStrides)
        val indicesArray = allocateNDArray(DataType.LONG, outputStrides) as MutableLongNDArray

        val countIterations = shape.sliceArray(0 until actualAxis).fold(1) { acc, i -> acc * i }
        val countElements = shape.sliceArray((actualAxis + 1) until rank).fold(1) { acc, i -> acc * i }
        val countDims = shape[actualAxis]


        val inputPointer = this.array.pointer()
        val outputPointer = outputArray.array.pointer()
        val outputIndicesPointer = indicesArray.array.pointer()

        when {
            k == 1 && actualAxis == shape.lastIndex -> {
                if (largest) {
                    repeat(countIterations) {
                        var maximum = inputPointer.getAndIncrement()
                        var maximumIndex = 0

                        inputPointer.forEachIndexed(countDims - 1, 1) { index: Int, value: PrimitiveType ->
                            if (value > maximum)  {
                                maximum = value
                                maximumIndex = index
                            }
                        }

                        outputPointer.set(maximum)
                        outputPointer.increment()

                        outputIndicesPointer.set(maximumIndex.toLong())
                        outputIndicesPointer.increment()
                    }
                } else {
                    repeat(countIterations) {
                        var minimum = inputPointer.getAndIncrement()
                        var minimumIndex = 0

                        inputPointer.forEachIndexed(countDims - 1, 1) { index: Int, value: PrimitiveType ->
                            if (value < minimum)  {
                                minimum = value
                                minimumIndex = index
                            }
                        }

                        outputPointer.set(minimum)
                        outputPointer.increment()

                        outputIndicesPointer.set(minimumIndex.toLong())
                        outputIndicesPointer.increment()
                    }
                }
            }

            k == 1 -> {
                if (largest) {
                    repeat(countIterations) { iteration ->
                        outputPointer.linearIndex = iteration * countElements
                        outputPointer.accept(inputPointer, countElements) { _: PrimitiveType, src: PrimitiveType -> src }
                        for (dim in 1 until countDims) {
                            outputPointer.linearIndex = iteration * countElements
                            outputIndicesPointer.linearIndex = iteration * countElements

                            var end = countElements

                            if (inputPointer.isCompatibleWith(outputPointer) && inputPointer.isCompatibleWith(outputIndicesPointer)) {
                                while (end > 0) {
                                    val outputBlock = outputPointer.currentBlock
                                    val offset = outputPointer.indexInBlock
                                    outputPointer.blockIncrement()

                                    val inputBlock = inputPointer.currentBlock
                                    inputPointer.blockIncrement()

                                    val outputIndicesBlock = outputIndicesPointer.currentBlock
                                    outputIndicesPointer.blockIncrement()

                                    for (index in offset until min(outputBlock.size, offset + end)) {
                                        val inputValue = inputBlock[index]
                                        val outputValue = outputBlock[index]

                                        if (inputValue > outputValue) {
                                            outputBlock[index] = inputValue
                                            outputIndicesBlock[index] = dim.toLong()
                                        }
                                    }

                                    end -= outputBlock.size - offset
                                }
                            } else {
                                while (end > 0) {
                                    val inputValue = inputPointer.getAndIncrement()
                                    val outputValue = outputPointer.get()

                                    if (inputValue > outputValue) {
                                        outputPointer.set(inputValue)
                                        outputIndicesPointer.set(dim.toLong())
                                    }

                                    outputPointer.increment()
                                    outputIndicesPointer.increment()
                                    end--
                                }
                            }
                        }
                    }
                } else {
                    repeat(countIterations) { iteration ->
                        outputPointer.linearIndex = iteration * countElements
                        outputPointer.accept(inputPointer, countElements) { _: PrimitiveType, src: PrimitiveType -> src }
                        for (dim in 1 until countDims) {
                            outputPointer.linearIndex = iteration * countElements
                            outputIndicesPointer.linearIndex = iteration * countElements

                            var end = countElements

                            if (inputPointer.isCompatibleWith(outputPointer) && inputPointer.isCompatibleWith(outputIndicesPointer)) {
                                while (end > 0) {
                                    val outputBlock = outputPointer.currentBlock
                                    val offset = outputPointer.indexInBlock
                                    outputPointer.blockIncrement()

                                    val inputBlock = inputPointer.currentBlock
                                    inputPointer.blockIncrement()

                                    val outputIndicesBlock = outputIndicesPointer.currentBlock
                                    outputIndicesPointer.blockIncrement()

                                    for (index in offset until min(outputBlock.size, offset + end)) {
                                        val inputValue = inputBlock[index]
                                        val outputValue = outputBlock[index]

                                        if (inputValue < outputValue) {
                                            outputBlock[index] = inputValue
                                            outputIndicesBlock[index] = dim.toLong()
                                        }
                                    }

                                    end -= outputBlock.size - offset
                                }
                            } else {
                                while (end > 0) {
                                    val inputValue = inputPointer.getAndIncrement()
                                    val outputValue = outputPointer.get()

                                    if (inputValue < outputValue) {
                                        outputPointer.set(inputValue)
                                        outputIndicesPointer.set(dim.toLong())
                                    }

                                    outputPointer.increment()
                                    outputIndicesPointer.increment()
                                    end--
                                }
                            }
                        }
                    }
                }
            }

            actualAxis == shape.lastIndex -> {
                if (largest) {
                    val maxHeap = PrimitiveMaxHeap(k)
                    repeat(countIterations) {
                        inputPointer.forEachIndexed(k) { index: Int, value: PrimitiveType -> maxHeap.insert(value, index) }

                        inputPointer.forEachIndexed(countDims - k, k) { index: Int, value: PrimitiveType ->
                            if (value > maxHeap.minValue) {
                                maxHeap.removeMin()
                                maxHeap.insert(value, index)
                            }
                        }

                        val (values, indices) = if (sorted) maxHeap.sorted() else maxHeap.data to maxHeap.indices

                        var outputIndex = 0
                        outputPointer.map(k) { values[outputIndex++] }

                        outputIndex = 0
                        outputIndicesPointer.map(k) { indices[outputIndex++].toLong() }

                        maxHeap.clear()
                    }
                } else {
                    val minHeap = PrimitiveMinHeap(k)
                    repeat(countIterations) {
                        inputPointer.forEachIndexed(k) { index: Int, value: PrimitiveType -> minHeap.insert(value, index) }

                        inputPointer.forEachIndexed(countDims - k, k) { index: Int, value: PrimitiveType ->
                            if (value < minHeap.maxValue) {
                                minHeap.removeMax()
                                minHeap.insert(value, index)
                            }
                        }

                        val (values, indices) = if (sorted) minHeap.sorted() else minHeap.data to minHeap.indices

                        var outputIndex = 0
                        outputPointer.map(k) { values[outputIndex++] }

                        outputIndex = 0
                        outputIndicesPointer.map(k) { indices[outputIndex++].toLong() }

                        minHeap.clear()
                    }
                }
            }

            else -> {
                if (largest) {
                    val maxHeaps = Array(countElements) { PrimitiveMaxHeap(k) }

                    repeat(countIterations) {
                        repeat(k) { dim ->
                            inputPointer.forEachIndexed(countElements) { index: Int, value: PrimitiveType -> maxHeaps[index].insert(value, dim) }
                        }

                        for (dim in k until countDims) {
                            inputPointer.forEachIndexed(countElements) { index: Int, value: PrimitiveType ->
                                val maxHeap = maxHeaps[index]
                                if (value > maxHeap.minValue) {
                                    maxHeap.removeMin()
                                    maxHeap.insert(value, dim)
                                }
                            }
                        }

                        val valuesAndIndicesArray = if (sorted) maxHeaps.map { it.sorted() } else maxHeaps.map { it.data to it.indices }

                        for (dim in 0 until k) {
                            var outputIndex = 0
                            outputPointer.map(countElements) { valuesAndIndicesArray[outputIndex++].first[dim] }

                            outputIndex = 0
                            outputIndicesPointer.map(countElements) { valuesAndIndicesArray[outputIndex++].second[dim].toLong() }
                        }

                        maxHeaps.forEach { it.clear() }
                    }
                } else {
                    val minHeaps = Array(countElements) { PrimitiveMinHeap(k) }
                    repeat(countIterations) {
                        repeat(k) { dim ->
                            inputPointer.forEachIndexed(countElements) { index: Int, value: PrimitiveType -> minHeaps[index].insert(value, dim) }
                        }

                        for (dim in k until countDims) {
                            inputPointer.forEachIndexed(countElements) { index: Int, value: PrimitiveType ->
                                val minHeap = minHeaps[index]
                                if (value < minHeap.maxValue) {
                                    minHeap.removeMax()
                                    minHeap.insert(value, dim)
                                }
                            }
                        }

                        val valuesAndIndicesArray = if (sorted) minHeaps.map { it.sorted() } else minHeaps.map { it.data to it.indices }

                        for (dim in 0 until k) {
                            var outputIndex = 0
                            outputPointer.map(countElements) { valuesAndIndicesArray[outputIndex++].first[dim] }

                            outputIndex = 0
                            outputIndicesPointer.map(countElements) { valuesAndIndicesArray[outputIndex++].second[dim].toLong() }
                        }

                        minHeaps.forEach { it.clear() }
                    }
                }
            }
        }

        return outputArray to indicesArray
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutablePrimitiveNDArray(array.copyOf(), strides)
    }

    @FilterPrimitives(exclude = [DataType.DOUBLE, DataType.FLOAT, DataType.BOOLEAN, DataType.BOOLEAN, DataType.INT, DataType.LONG, DataType.SHORT,
        DataType.UINT, DataType.ULONG, DataType.USHORT])
    @BindPrimitives(type1 = [DataType.BYTE, DataType.UBYTE])
    fun quantizeDot(other: @BindPrimitives.Type1 PrimitiveNDArray, destination: MutableFloatNDArray, zeroPointA: Int = 0, zeroPointB: Int = 0, scale: Float = 1f): MutableFloatNDArray {
        val M = this.shape[0]

        fun wrapper(body: (inner: () -> Unit) -> Unit = { it() }) {
            for (rdBlockNum in 0 until destination.blocksInRow) {
                body {
                    for (i in 0 until M) {
                        val dBlockOffset = i * destination.blocksInRow
                        val lBlockOffset = i * this.blocksInRow

                        var k = 0
                        for (lBlockNum in 0 until this.blocksInRow) {
                            val lBlock = this.array.blocks[lBlockOffset + lBlockNum]
                            for (lInd in lBlock.indices) {
                                val temp = lBlock[lInd].toInt() - zeroPointA
                                val rBlockOffset = k * other.blocksInRow
                                val rBlock = other.array.blocks[rBlockOffset + rdBlockNum]
                                val dBlock = destination.array.blocks[dBlockOffset + rdBlockNum]
                                for (idx in rBlock.indices) {
                                    dBlock[idx] += (temp * (rBlock[idx].toInt() - zeroPointB)) * scale
                                }
                                k++
                            }
                        }
                    }
                }
            }
        }

        if (other.blocksInRow > 1) {
            runBlocking(Dispatchers.Default) { wrapper { launch { it() } } }
        } else {
            wrapper()
        }

        return destination
    }

    override fun expand(shape: IntArray): MutablePrimitiveNDArray {
        val outputShape = Broadcasting.broadcastShape(listOf(this.shape, shape))
        val output = allocateNDArray(Strides(outputShape))
        Broadcasting.applyWithBroadcast(listOf(this), output) { inputs: List<NDArray>, destination: MutableNDArray ->
            destination as MutablePrimitiveNDArray
            val input = inputs[0] as PrimitiveNDArray
            destination.copyFrom(0, input)
        }

        return output
    }

    override fun nonZero(): LongNDArray {
        if (isScalar()) {
            val value = singleValue()
            return if (value != (0).toPrimitive())
                LongNDArray(LongTiledArray(emptyArray()), Strides(intArrayOf(1, 0)))
            else
                LongNDArray(Strides(intArrayOf(1, 1))) { 0L }
        }
        val ndIndexSize = shape.size
        var totalElements = 0
        val inputPointer = array.pointer()
        val indicesArray = LongArray(linearSize * ndIndexSize)
        this.ndIndexed { ndIndex ->
            if (inputPointer.getAndIncrement() != (0).toPrimitive()) {
                ndIndex.copyInto(indicesArray, totalElements * ndIndexSize)
                totalElements++
            }
        }
        val nonZeroStrides = Strides(intArrayOf(ndIndexSize, totalElements))
        val indicesByDim = LongTiledArray(nonZeroStrides)
        val resultPointer = indicesByDim.pointer()
        for (i in 0 until ndIndexSize)
            for (j in 0 until totalElements) {
                resultPointer.set(indicesArray[j * ndIndexSize + i])
                resultPointer.increment()
            }
        return LongNDArray(indicesByDim, nonZeroStrides)
    }

    override fun pad(pads: Array<Pair<Int, Int>>, mode: String, constantValue: NDArray?): PrimitiveNDArray {
        require(pads.size == rank)
        val outputShape = shape.copyOf()
        for ((axis, pad) in pads.withIndex()) {
            outputShape[axis] += pad.first + pad.second
        }

        val outputArray = allocateNDArray(Strides(outputShape))
        val constant = (constantValue?.singleValue() ?: (0).toPrimitive()) as PrimitiveType

        fun recurrentCopyInput(axis: Int, input: PrimitiveNDArray, output: MutablePrimitiveNDArray) {
            val leftPad = pads[axis].first
            val inputDims = input.shape[0]

            if (axis == this.rank - 1) {
                output.copyFrom(leftPad, input)
                return
            }

            for (dim in 0 until inputDims) {
                recurrentCopyInput(axis + 1, input.view(dim), output.viewMutable(dim + leftPad))
            }
        }

        fun recurrentFillConstant(axis: Int, output: MutablePrimitiveNDArray) {
            val (leftPad, rightPad) = pads[axis]
            val outputDims = output.shape[0]

            if (axis == this.rank - 1) {
                output.fill(constant, from = 0, to = leftPad)
                output.fill(constant, from = outputDims - rightPad, to = outputDims)
                return
            }

            for (dim in 0 until leftPad) {
                output.viewMutable(dim).fill(constant)
            }
            for (dim in leftPad until outputDims - rightPad) {
                recurrentFillConstant(axis + 1, output.viewMutable(dim))
            }
            for (dim in outputDims - rightPad until outputDims) {
                output.viewMutable(dim).fill(constant)
            }
        }

        fun recurrentFillEdge(axis: Int, output: MutablePrimitiveNDArray) {
            val (leftPad, rightPad) = pads[axis]
            val outputDims = output.shape[0]

            if (axis == this.rank - 1) {
                val leftPadValue = output.array[leftPad]
                val rightPadValue = output.array[outputDims - rightPad - 1]
                output.fill(leftPadValue, from = 0, to = leftPad)
                output.fill(rightPadValue, from = outputDims - rightPad, to = outputDims)
                return
            }

            for (dim in leftPad until outputDims - rightPad) {
                recurrentFillEdge(axis + 1, output.viewMutable(dim))
            }

            val leftPadArray = output.view(leftPad)
            for (dim in 0 until leftPad) {
                output.viewMutable(dim).copyFrom(offset = 0, leftPadArray)
            }

            val rightPadArray = output.view(outputDims - rightPad - 1)
            for (dim in outputDims - rightPad until outputDims) {
                output.viewMutable(dim).copyFrom(offset = 0, rightPadArray)
            }
        }

        fun recurrentFillReflect(axis: Int, output: MutablePrimitiveNDArray) {
            val (leftPad, rightPad) = pads[axis]
            val outputDims = output.shape[0]

            if (axis == this.rank - 1) {
                val leftPadInputPointer = output.array.pointer(leftPad + 1)
                val leftPadOutputPointer = output.array.pointer(leftPad - 1)

                repeat(leftPad) {
                    if (leftPadInputPointer.linearIndex == outputDims - rightPad) {
                        leftPadInputPointer.linearIndex = leftPad
                    }
                    leftPadOutputPointer.set(leftPadInputPointer.getAndIncrement())
                    leftPadOutputPointer.decrement()
                }

                val rightPadInputPointer = output.array.pointer(outputDims - rightPad - 2)
                val rightPadOutputPointer = output.array.pointer(outputDims - rightPad)

                repeat(rightPad) {
                    if (rightPadInputPointer.linearIndex == leftPad - 1) {
                        rightPadInputPointer.linearIndex = outputDims - rightPad - 1
                    }
                    rightPadOutputPointer.set(rightPadInputPointer.get())
                    rightPadInputPointer.decrement()
                    rightPadOutputPointer.increment()
                }

                return
            }

            for (dim in leftPad until outputDims - rightPad) {
                recurrentFillReflect(axis + 1, output.viewMutable(dim))
            }

            var leftPadInputAxis = leftPad + 1
            var leftPadOutputAxis = leftPad - 1
            repeat(leftPad) {
                if (leftPadInputAxis == outputDims - rightPad) {
                    leftPadInputAxis = leftPad
                }
                output.viewMutable(leftPadOutputAxis).copyFrom(offset = 0, output.view(leftPadInputAxis))
                leftPadInputAxis++
                leftPadOutputAxis--
            }

            var rightPadInputAxis = outputDims - rightPad - 2
            var rightPadOutputAxis = outputDims - rightPad

            repeat(rightPad) {
                if (rightPadInputAxis == leftPad - 1) {
                    rightPadInputAxis = outputDims - rightPad - 1
                }

                output.viewMutable(rightPadOutputAxis).copyFrom(offset = 0, output.view(rightPadInputAxis))
                rightPadInputAxis--
                rightPadOutputAxis++
            }
        }

        recurrentCopyInput(0, this, outputArray)

        when(mode) {
            "constant" -> {
                if (constant != (0).toPrimitive()) {
                    recurrentFillConstant(0, outputArray)
                }
            }
            "edge" -> {
                recurrentFillEdge(0, outputArray)
            }
            "reflect" -> {
                recurrentFillReflect(0, outputArray)
            }
            else -> error("Unsupported mode")
        }

        return outputArray
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PrimitiveNDArray) return false

        if (type != other.type) return false
        if (strides != other.strides) return false
        if (array != other.array) return false

        return true
    }

    override fun hashCode(): Int {
        var result = array.hashCode()
        result = 31 * result + strides.hashCode()
        result = 31 * result + type.hashCode()
        return result
    }

    companion object {
        fun scalar(value: PrimitiveType): PrimitiveNDArray {
            return PrimitiveNDArray(PrimitiveTiledArray(1, 1) { value }, Strides.EMPTY)
        }
    }
}

@GenerateNameFromPrimitives
interface PrimitiveMap : PrimitiveToPrimitiveFunction {
    fun apply(value: PrimitiveType): PrimitiveType
}
