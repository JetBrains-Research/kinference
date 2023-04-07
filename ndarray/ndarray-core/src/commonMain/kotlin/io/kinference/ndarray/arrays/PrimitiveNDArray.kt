@file:GeneratePrimitives(DataType.NUMBER)
@file:Suppress("DuplicatedCode", "unused")

package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.dot.DotUtils
import io.kinference.ndarray.extensions.dot.dotParallelM
import io.kinference.ndarray.extensions.dot.dotParallelN
import io.kinference.ndarray.extensions.softmax.softmax
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlin.jvm.JvmName
import kotlin.math.*

@GenerateNameFromPrimitives
open class PrimitiveNDArray(array: PrimitiveTiledArray, strides: Strides) : NumberNDArrayCore {
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

    override operator fun get(index: IntArray): PrimitiveType {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        return array[linearIndex]
    }

    override fun getLinear(index: Int): PrimitiveType {
        return array[index]
    }

    override fun singleValue(): PrimitiveType {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array.blocks[0][0]
    }

    override fun clone(): PrimitiveNDArray {
        return PrimitiveNDArray(array.copyOf(), Strides(shape))
    }

    override fun toMutable(): MutablePrimitiveNDArray = MutablePrimitiveNDArray(array.copyOf(), strides)

    override suspend fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutablePrimitiveNDArray {
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

    override suspend fun concat(others: List<NDArray>, axis: Int): MutablePrimitiveNDArray {
        val actualAxis = indexAxis(axis)

        val inputs = listOf(this) + others
        val resultShape = shape.copyOf()
        resultShape[actualAxis] = inputs.sumOf { it.shape[actualAxis] }

        val result = MutablePrimitiveNDArray(Strides(resultShape))
        val resultPointer = result.array.pointer()

        val numIterations = resultShape.take(actualAxis).fold(1, Int::times)
        val pointersToSteps = inputs.map {
            require(it is PrimitiveNDArray)
            it.array.pointer() to it.shape.drop(actualAxis).fold(1, Int::times)
        }

        repeat(numIterations) {
            pointersToSteps.forEach { (pointer, numSteps) ->
                resultPointer.accept(pointer, numSteps) { _, src -> src }
            }
        }
        return result
    }

    override suspend fun map(function: PrimitiveToPrimitiveFunction) = map(function, MutablePrimitiveNDArray(strides))

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutablePrimitiveNDArray {
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

    override suspend fun min(): PrimitiveType {
        var min = PrimitiveType.MAX_VALUE
        for (block in array.blocks) {
            for (idx in block.indices) {
                val tmp = block[idx]
                if (tmp < min) min = tmp
            }
        }
        return min
    }

    override suspend fun min(axis: Int, keepDims: Boolean): PrimitiveNDArray {
        return findComparable(axis, keepDims) { first, second -> first < second }
    }

    override suspend fun max(): PrimitiveType {
        var max = PrimitiveType.MIN_VALUE
        for (block in array.blocks) {
            for (idx in block.indices) {
                val tmp = block[idx]
                if (tmp > max) max = tmp
            }
        }

        return max
    }

    override suspend fun sum(): PrimitiveType {
        var sum = (0).toPrimitive()

        for (block in array.blocks) {
            for (idx in block.indices) {
                sum = (sum + block[idx]).toPrimitive()
            }
        }
        return sum
    }

    override suspend fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutablePrimitiveNDArray {
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

    override suspend fun erf(): PrimitiveNDArray {
        return this.map(object : PrimitiveMap {
            override fun apply(value: PrimitiveType): PrimitiveType = erf(value)
        })
    }

    override suspend fun softmax(axis: Int): PrimitiveNDArray {
        return softmax(this, axis, strides) as PrimitiveNDArray
    }

    override suspend fun logSoftmax(axis: Int): PrimitiveNDArray {
        fun log(type: DataType) = when(type) {
            DataType.FLOAT -> object : FloatMap {
                override fun apply(value: Float): Float = ln(value)
            }

            DataType.DOUBLE -> object : DoubleMap {
                override fun apply(value: Double): Double = ln(value)
            }
            else -> error("LogSoftmax supported only for DOUBLE and FLOAT types")
        }
        val output = softmax(this, axis, strides)
        return output.mapMutable(log(output.type)) as MutablePrimitiveNDArray
    }

    override suspend fun plus(other: NumberNDArray): MutablePrimitiveNDArray {
        val destShape = broadcastShape(listOf(this.shape, other.shape))
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

    override suspend fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutablePrimitiveNDArray {
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

    override suspend fun minus(other: NumberNDArray): MutablePrimitiveNDArray {
        val destShape = broadcastShape(listOf(this.shape, other.shape))
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

    override suspend fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutablePrimitiveNDArray {
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

    override suspend fun times(other: NumberNDArray): MutablePrimitiveNDArray {
        val destShape = broadcastShape(listOf(this.shape, other.shape))
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

    override suspend fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutablePrimitiveNDArray {
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

    override suspend fun div(other: NumberNDArray): MutablePrimitiveNDArray {
        val destShape = broadcastShape(listOf(this.shape, other.shape))
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

    override suspend fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutablePrimitiveNDArray {
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

    override suspend fun dot(other: NumberNDArray, destination: MutableNumberNDArray): MutablePrimitiveNDArray {
        other as PrimitiveNDArray; destination as MutablePrimitiveNDArray
        require(shape.size in 1..2 && other.shape.size in 1..2)
        val actualThis = (if (this.shape.size == 1) this.reshape(intArrayOf(1, shape[0])) else this) as PrimitiveNDArray
        val actualOther = (if (other.shape.size == 1) other.reshape(intArrayOf(1, other.shape[0])) else other) as PrimitiveNDArray

        require(actualThis.shape[1] == actualOther.shape[0])

        val n = actualThis.shape[0]
        val t = actualThis.shape[1]
        val m = actualOther.shape[1]

        val rdBlocksInRow = actualOther.blocksInRow

        val rdBlockSize = actualOther.array.blockSize

        val countCoroutinesForNParallelization = countCoroutinesByData(t * m, n, DotUtils.MIN_DATA_PER_LAUNCH)
        val countCoroutinesForMParallelization = countCoroutinesByData(rdBlockSize * n * t, rdBlocksInRow, DotUtils.MIN_DATA_PER_LAUNCH)

        return when {
            countCoroutinesForNParallelization > 1 -> dotParallelN(actualThis, actualOther, destination)
            countCoroutinesForMParallelization > 1 -> dotParallelM(actualThis, actualOther, destination)
            // Fallback without parallelization
            else -> dotParallelN(actualThis, actualOther, destination)
        }
    }


    override suspend fun dot(other: NumberNDArray): MutablePrimitiveNDArray {
        other as PrimitiveNDArray
        require(shape.size in 1..2 && other.shape.size in 1..2)

        val destination = MutablePrimitiveNDArray(intArrayOf(shape[0], other.shape[1]))
        return dot(other, destination)
    }

    override suspend fun matmul(other: NumberNDArray): MutablePrimitiveNDArray {
        other as NumberNDArrayCore
        val outputShape = Broadcasting.broadcastShapeForMatmul(this.shape, other.shape)
        val outputArray = MutablePrimitiveNDArray(outputShape)
        return matmul(other, outputArray) { otherArray, dest -> this.dot(otherArray, dest) } as MutablePrimitiveNDArray
    }

    override suspend fun matmul(other: NumberNDArray, destination: MutableNumberNDArrayCore): MutablePrimitiveNDArray {
        other as NumberNDArrayCore
        return matmul(other, destination) { otherArray, dest -> this.dot(otherArray, dest) } as MutablePrimitiveNDArray
    }

    override suspend fun gemm(
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        lda: Int,
        b: NDArray,
        ldb: Int,
        beta: Double,
        c: MutableNDArray,
        ldc: Int,
        aOffset: Int,
        bOffset: Int,
        cOffset: Int,
        transposeA: Boolean,
        transposeB: Boolean
    ): MutablePrimitiveNDArray {
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

    override suspend fun argmax(axis: Int, keepDims: Boolean, selectLastIndex: Boolean): IntNDArray {
        val actualAxis = indexAxis(axis)

        val countIterations = computeBlockSize(toDim = actualAxis)
        val countElements = computeBlockSize(fromDim = actualAxis + 1)
        val countDims = shape[actualAxis]

        val outputShape = if (keepDims) shape.copyOf().apply { set(actualAxis, 1) } else shape.sliceArray(shape.indices.minus(actualAxis))
        val outputArray = MutableIntNDArray(outputShape)

        val inputPointer = this.array.pointer()

        if (actualAxis == shape.lastIndex || countElements == 1) {
            val outputPointer = outputArray.array.pointer()
            for (i in 0 until countIterations) {
                var maxValue = inputPointer.getAndIncrement()
                var maxIndex = 0

                if (selectLastIndex) {
                    inputPointer.forEachIndexed(countDims - 1) { index: Int, value: PrimitiveType ->
                        if (value >= maxValue) {
                            maxValue = value
                            maxIndex = index + 1
                        }
                    }
                } else {
                    inputPointer.forEachIndexed(countDims - 1) { index: Int, value: PrimitiveType ->
                        if (value > maxValue) {
                            maxValue = value
                            maxIndex = index + 1
                        }
                    }
                }
                outputPointer.set(maxIndex)
                outputPointer.increment()
            }

            return outputArray
        }

        val tempMaxValues = PrimitiveTiledArray(countElements, outputArray.array.blockSize)

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

                        if (selectLastIndex) {
                            for (index in offset until min(outputBlock.size, offset + end)) {
                                val value = inputBlock[index]
                                if (value >= maxValuesBlock[index]) {
                                    maxValuesBlock[index] = value
                                    outputBlock[index] = j
                                }
                            }
                        } else {
                            for (index in offset until min(outputBlock.size, offset + end)) {
                                val value = inputBlock[index]
                                if (value > maxValuesBlock[index]) {
                                    maxValuesBlock[index] = value
                                    outputBlock[index] = j
                                }
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

    private fun findComparable(axis: Int, keepDims: Boolean, compare: (PrimitiveType, PrimitiveType) -> Boolean): PrimitiveNDArray {
        val actualAxis = indexAxis(axis)

        val countIterations = computeBlockSize(toDim = actualAxis)
        val countElements = computeBlockSize(fromDim = actualAxis + 1)
        val countDims = shape[actualAxis]

        val outputShape = if (keepDims) shape.copyOf().apply { set(actualAxis, 1) } else shape.sliceArray(shape.indices.minus(actualAxis))
        val outputArray = PrimitiveNDArray(Strides(outputShape))

        val inputPointer = this.array.pointer()

        if (actualAxis == shape.lastIndex || countElements == 1) {
            val outputPointer = outputArray.array.pointer()
            for (i in 0 until countIterations) {
                var result = inputPointer.getAndIncrement()
                inputPointer.forEach(countDims - 1) { value: PrimitiveType ->
                    if (compare(value, result)) {
                        result = value
                    }
                }
                outputPointer.set(result)
                outputPointer.increment()
            }

            return outputArray
        }

        for (i in 0 until countIterations) {
            val outputPointer = outputArray.array.pointer(i * countElements)
            outputPointer.accept(inputPointer, countElements) { _: PrimitiveType, src: PrimitiveType -> src }

            for (j in 1 until countDims) {
                outputPointer.linearIndex = i * countElements
                outputPointer.accept(inputPointer, countElements) { dst: PrimitiveType, src: PrimitiveType ->
                    if (compare(src, dst))
                        src
                    else
                        dst
                }
            }
        }

        return outputArray
    }

    override suspend fun max(axis: Int, keepDims: Boolean): PrimitiveNDArray {
        return findComparable(axis, keepDims) { first, second -> first > second }
    }

    override suspend fun reduceSum(axes: IntArray, keepDims: Boolean): PrimitiveNDArray {
        val axesToReduce = axes.map { indexAxis(it) }.toSet()
        require(axesToReduce.all { it in shape.indices }) { "Axes ${axes.joinToString()} must be in range [-$rank, ${rank - 1}]" }

        val outputShapeWithKeepDims = shape.copyOf().apply { axesToReduce.forEach { set(it, 1) } }
        val stridesWithKeepDims = Strides(outputShapeWithKeepDims)

        val outputShape = if (keepDims) outputShapeWithKeepDims else shape.sliceArray(shape.indices.minus(axesToReduce))
        val outputArray = PrimitiveNDArray(Strides(outputShape))

        val axisToStop = axesToReduce.maxOrNull()!! + 1
        val blockToApply = computeBlockSize(fromDim = axisToStop)


        fun reduceSumRecurrent(axis: Int, inputOffset: Int, outputOffset: Int) {
            when(axis) {
                axisToStop -> {
                    val inputPointer = this.array.pointer(inputOffset)
                    val outputPointer = outputArray.array.pointer(outputOffset)

                    outputPointer.accept(inputPointer, blockToApply) { dst: PrimitiveType, src: PrimitiveType -> (dst + src).toPrimitive() }
                }
                shape.lastIndex -> {
                    val dim = this.shape[axis]
                    val inputPointer = this.array.pointer(inputOffset)
                    val outputPointer = outputArray.array.pointer(outputOffset)

                    var accumulator = outputPointer.get()
                    inputPointer.forEach(dim) { accumulator = (accumulator + it).toPrimitive() }
                    outputPointer.set(accumulator)
                }
                else -> {
                    val dim = this.shape[axis]
                    repeat(dim) {
                        val inputAdditionalOffset = this.strides.strides[axis] * it
                        val outputAdditionalOffset = if (axis in axesToReduce) 0 else stridesWithKeepDims.strides[axis] * it

                        reduceSumRecurrent(axis + 1, inputOffset + inputAdditionalOffset, outputOffset + outputAdditionalOffset)
                    }
                }
            }
        }

        reduceSumRecurrent(0, 0, 0)

        return outputArray
    }

    override suspend fun topK(axis: Int, k: Int, largest: Boolean, sorted: Boolean): Pair<PrimitiveNDArray, LongNDArray> {
        val actualAxis = indexAxis(axis)
        val outputStrides = Strides(shape.copyOf().apply { set(actualAxis, k) })

        val outputArray = PrimitiveNDArray(outputStrides)
        val indicesArray = MutableLongNDArray(outputStrides)

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

    override fun copyIfNotMutable(): MutablePrimitiveNDArray {
        return MutablePrimitiveNDArray(array.copyOf(), strides)
    }

    override suspend fun expand(shape: IntArray): MutablePrimitiveNDArray {
        val outputShape = broadcastShape(listOf(this.shape, shape))
        val output = MutablePrimitiveNDArray(Strides(outputShape))
        Broadcasting.applyWithBroadcast(listOf(this), output) { inputs: List<NDArray>, destination: MutableNDArray ->
            destination as MutablePrimitiveNDArray
            val input = inputs[0] as PrimitiveNDArray
            destination.copyFrom(0, input)
        }

        return output
    }

    override suspend fun nonZero(): LongNDArray {
        if (isScalar()) {
            val value = singleValue()
            return if (value == (0).toPrimitive())
                LongNDArray(LongTiledArray(emptyArray()), Strides(intArrayOf(0, 1)))
            else
                LongNDArray(Strides(intArrayOf(0, 1))) { 0L }
        }
        val ndIndexSize = rank
        var totalElements = 0
        val inputPointer = array.pointer()
        val indicesArray = IntArray(linearSize * ndIndexSize)
        this.ndIndices { ndIndex ->
            if (inputPointer.getAndIncrement() != (0).toPrimitive()) {
                ndIndex.copyInto(indicesArray, totalElements * ndIndexSize)
                totalElements++
            }
        }
        return LongNDArray(intArrayOf(ndIndexSize, totalElements)) { (i, j): IntArray ->
            indicesArray[j * ndIndexSize + i].toLong()
        }
    }

    override suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): PrimitiveNDArray {
        require(pads.size == rank)
        val outputShape = shape.copyOf()
        for ((axis, pad) in pads.withIndex()) {
            outputShape[axis] += pad.first + pad.second
        }

        val outputArray = MutablePrimitiveNDArray(Strides(outputShape))
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
            PadMode.CONSTANT -> {
                if (constant != (0).toPrimitive()) {
                    recurrentFillConstant(0, outputArray)
                }
            }
            PadMode.EDGE -> {
                recurrentFillEdge(0, outputArray)
            }
            PadMode.REFLECT -> {
                recurrentFillReflect(0, outputArray)
            }
        }

        return outputArray
    }

    override suspend fun tile(repeats: IntArray): PrimitiveNDArray {
        require(repeats.size == rank)

        if (repeats.all { it == 1 }) return this.toMutable()

        val outputShape = shape.copyOf().apply {
            for (idx in indices) {
                this[idx] *= repeats[idx]
            }
        }
        val outputArray = PrimitiveNDArray(Strides(outputShape))

        var axisToStop = -1
        for (idx in repeats.indices) {
            when {
                axisToStop == -1 && repeats[idx] == 1 -> axisToStop = idx
                repeats[idx] != 1 -> axisToStop = -1
            }
        }

        val blockToCopy = if (axisToStop != -1) computeBlockSize(fromDim = axisToStop) else 0

        fun tileCopy(axis: Int, inputOffset: Int, outputOffset: Int) {
            when(axis) {
                axisToStop -> {
                    val inputPointer = this.array.pointer(inputOffset)
                    val outputPointer = outputArray.array.pointer(outputOffset)

                    outputPointer.accept(inputPointer, blockToCopy) { _: PrimitiveType, src: PrimitiveType -> src }
                }

                shape.lastIndex -> {
                    val inputPointer = this.array.pointer(inputOffset)
                    val outputPointer = outputArray.array.pointer(outputOffset)

                    outputPointer.accept(inputPointer, shape.last()) { _: PrimitiveType, src: PrimitiveType -> src }
                }

                else -> {
                    val dims = this.shape[axis]

                    repeat(dims) { dim ->
                        val additionalInputOffset = dim * this.strides.strides[axis]
                        val additionalOutputOffset = dim * outputArray.strides.strides[axis]

                        tileCopy(axis + 1, inputOffset + additionalInputOffset, outputOffset + additionalOutputOffset)
                    }
                }
            }
        }

        fun tileRepeat(axis: Int, offset: Int) {
            val countRepeat = repeats[axis]

            when(axis) {
                axisToStop -> return
                shape.lastIndex -> {
                    val blockSize = this.shape[axis]
                    val inputPointer = outputArray.array.pointer()
                    val outputPointer = outputArray.array.pointer(offset + blockSize)

                    repeat(countRepeat - 1) {
                        inputPointer.linearIndex = offset
                        outputPointer.accept(inputPointer, blockSize) { _: PrimitiveType, src: PrimitiveType -> src }
                    }
                }
                else -> {
                    val dims = this.shape[axis]
                    repeat(dims) { dim ->
                        val additionalOffset = dim * outputArray.strides.strides[axis]
                        tileRepeat(axis + 1, offset + additionalOffset)
                    }

                    if (countRepeat > 1) {
                        val blockSize = outputArray.strides.strides[axis] * this.shape[axis]
                        val inputPointer = outputArray.array.pointer()
                        val outputPointer = outputArray.array.pointer(offset + blockSize)
                        repeat(countRepeat - 1) {
                            inputPointer.linearIndex = offset
                            outputPointer.accept(inputPointer, blockSize) { _: PrimitiveType, src: PrimitiveType -> src }
                        }
                    }
                }
            }
        }

        tileCopy(0, 0, 0)
        tileRepeat(0, 0)

        return outputArray
    }

    override suspend fun reshape(strides: Strides): PrimitiveNDArray {
        require(strides.linearSize == this.strides.linearSize) { "Linear size must be equal" }

        if (strides.shape.isNotEmpty() && this.shape.isNotEmpty() && strides.shape.last() != this.shape.last()) {
            val newArray = PrimitiveTiledArray(strides)
            this.array.copyInto(newArray)

            return PrimitiveNDArray(newArray, strides)
        }

        return MutablePrimitiveNDArray(this.array, strides)
    }

    private fun transposeByBlocks(permutations: IntArray): PrimitiveNDArray {
        val outputBlocks =  this.array.blocks.copyOf()
        val outputStrides = strides.transpose(permutations)

        var axisToStop: Int = permutations.size

        for (idx in permutations.indices.reversed()) {
            if (permutations[idx] != idx) {
                axisToStop = idx + 1
                break
            }
        }
        val countBlocksToCopy = computeBlockSize(fromDim = axisToStop) / this.array.blockSize


        fun transposeByBlocksRec(axis: Int, inputOffset: Int, outputOffset: Int) {
            when(axis) {
                shape.lastIndex, axisToStop -> {
                    val inputStartBlockNum = inputOffset / this.array.blockSize
                    val outputStartBlockNum = outputOffset / this.array.blockSize

                    repeat(countBlocksToCopy) {
                        outputBlocks[outputStartBlockNum + it] = this.array.blocks[inputStartBlockNum + it]
                    }
                }

                else -> {
                    val dims = outputStrides.shape[axis]

                    repeat(dims) { dim ->
                        val additionalInputOffset = this.strides.strides[permutations[axis]] * dim
                        val additionalOutputOffset = outputStrides.strides[axis] * dim

                        transposeByBlocksRec(axis + 1, inputOffset + additionalInputOffset, outputOffset + additionalOutputOffset)
                    }
                }
            }
        }

        transposeByBlocksRec(0, 0, 0)

        return PrimitiveNDArray(PrimitiveTiledArray(outputBlocks), outputStrides)
    }

    override suspend fun transpose(permutations: IntArray): PrimitiveNDArray {
        require(permutations.size == rank)
        require(permutations.all { it in permutations.indices })

        if (permutations.withIndex().all { it.value == it.index }) {
            return this
        }

        val outputStrides = strides.transpose(permutations)

        if (isTransposeReshape(permutations)) {
            return reshape(outputStrides)
        }

        if (rank == 2) {
            return transpose2D()
        }

        if (permutations.lastIndex == permutations.last()) {
            return transposeByBlocks(permutations)
        }

        val outputArray = MutablePrimitiveNDArray(outputStrides)

        fun transposeRec(axis: Int, inputOffset: Int, outputOffset: Int) {
            when(axis) {
                shape.lastIndex -> {
                    val dims = outputStrides.shape[axis]
                    val inputStride = this.strides.strides[permutations[axis]]
                    var index = 0
                    outputArray.array.pointer(outputOffset).map(dims) { _: PrimitiveType -> this.array[inputOffset + inputStride * index++] }
                }

                else -> {
                    val dims = outputStrides.shape[axis]

                    repeat(dims) { dim ->
                        val inputAdditionalOffset = this.strides.strides[permutations[axis]] * dim
                        val outputAdditionalOffset = outputStrides.strides[axis] * dim

                        transposeRec(axis + 1, inputOffset + inputAdditionalOffset, outputOffset + outputAdditionalOffset)
                    }
                }
            }
        }

        transposeRec(0, 0, 0)

        return outputArray
    }

    override suspend fun transpose2D(): PrimitiveNDArray {
        require(rank == 2)

        val outputShape = shape.reversedArray()
        val outputStrides = Strides(outputShape)
        val outputArray = PrimitiveTiledArray(outputStrides)

        val newBlocksInRow = outputShape[1] / outputArray.blockSize

        var blockNum = 0

        for (row in 0 until outputShape[0]) {
            val (blockOffset, offset) = array.indexFor(row)
            var col = 0
            for (i in 0 until newBlocksInRow) {
                val block = outputArray.blocks[blockNum++]
                for (idx in 0 until outputArray.blockSize) {
                    block[idx] = this.array.blocks[blockOffset + col * this.blocksInRow][offset]
                    col++
                }
            }
        }

        return MutablePrimitiveNDArray(outputArray, outputStrides)
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
        fun zeros(shape: IntArray): MutablePrimitiveNDArray = MutablePrimitiveNDArray(shape)

        fun ones(shape: IntArray): MutablePrimitiveNDArray = MutablePrimitiveNDArray(shape) { 1.toPrimitive() }

        fun scalar(value: PrimitiveType): PrimitiveNDArray {
            return PrimitiveNDArray(PrimitiveTiledArray(1, 1) { value }, Strides.EMPTY)
        }

        operator fun invoke(strides: Strides, init: (IntArray) -> PrimitiveType): PrimitiveNDArray {
            val iterator = NDIndexer(strides)
            return PrimitiveNDArray(strides) { init(iterator.next()) }
        }

        operator fun invoke(shape: IntArray, init: (IntArray) -> PrimitiveType): PrimitiveNDArray {
            return invoke(Strides(shape), init)
        }

        operator fun invoke(vararg shape: Int): PrimitiveNDArray {
            return PrimitiveNDArray(PrimitiveTiledArray(shape), Strides(shape))
        }

        @JvmName("invokeNDVarArg")
        operator fun invoke(vararg shape: Int, init: (IntArray) -> PrimitiveType): PrimitiveNDArray {
            return invoke(Strides(shape), init)
        }

        @JvmName("invokeVarArg")
        operator fun invoke(vararg shape: Int, init: (Int) -> PrimitiveType): PrimitiveNDArray {
            return PrimitiveNDArray(PrimitiveTiledArray(shape, init), Strides(shape))
        }
    }
}

@GenerateNameFromPrimitives
interface PrimitiveMap : PrimitiveToPrimitiveFunction {
    fun apply(value: PrimitiveType): PrimitiveType
}
