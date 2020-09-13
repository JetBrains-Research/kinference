@file:GenerateWithPrimitives

package io.kinference.ndarray

import io.kinference.ndarray.extensions.*
import io.kinference.primitives.annotations.GenerateWithPrimitives
import io.kinference.primitives.annotations.PrimitiveClass
import io.kinference.primitives.types.*
import kotlin.math.*

@PrimitiveClass
@ExperimentalUnsignedTypes
open class PrimitiveNDArray(val array: PrimitiveArray, strides: Strides = Strides.empty(), override val offset: Int = 0) : NumberNDArray {
    override val type = DataType.UNKNOWN

    final override var strides: Strides = strides
        protected set

    override fun get(index: Int): PrimitiveType = array[index]
    override fun get(indices: IntArray): PrimitiveType = array[strides.offset(indices)]

    override fun allocateNDArray(strides: Strides): MutableNumberNDArray = MutablePrimitiveNDArray(PrimitiveArray(strides.linearSize), strides)

    override fun view(vararg axes: Int): NumberNDArray {
        val (additionalOffset, newShape) = viewHelper(axes, strides)
        return PrimitiveNDArray(array, Strides(newShape), offset + additionalOffset)
    }

    override fun toMutable(newStrides: Strides, additionalOffset: Int): MutableNumberNDArray = MutablePrimitiveNDArray(array.copyOfRange(offset + additionalOffset, offset + additionalOffset + newStrides.linearSize), newStrides)

    override fun map(function: PrimitiveToPrimitiveFunction): MutableNumberNDArray {
        function as PrimitiveMap
        val destination = allocateNDArray(strides) as MutablePrimitiveNDArray
        for (index in 0 until destination.linearSize) {
            destination.array[index] = function.apply(this.array[offset + index])
        }

        return destination
    }

    override fun erfFor(value: Any): PrimitiveType {
        value as PrimitiveType
        val sign = value.toDouble().sign
        val doubleValue = abs(value.toDouble())
        val t = 1 / (1 + ERF_P_VALUE * doubleValue)

        val sum = t * (ERF_COEF[0] + t * (ERF_COEF[1] + t * (ERF_COEF[2] + t * (ERF_COEF[3] + t * ERF_COEF[4]))))

        return (sign * (1.0 - sum * exp(- doubleValue * doubleValue))).toPrimitive()
    }

    override fun dequantize(zeroPoint: NDArray?, scale: NDArray, axis: Int): NDArray {
        scale as FloatNDArray
        val zeros = (zeroPoint as? PrimitiveNDArray)?.array
        val output = MutableFloatNDArray(FloatArray(this.linearSize), this.strides)
        when {
            canDequantizePerTensor(zeroPoint, scale) -> {
                val zero = zeros?.get(0)?.toFloat() ?: 0f
                for (i in 0 until output.linearSize) output[i] = (this[i].toFloat() - zero) * scale[0]
            }
            canDequantizePerAxis(axis, zeroPoint, scale) -> {
                val actualAxis = indexAxis(axis)
                val blockCount = computeBlockSize(toDim = actualAxis)
                val blockSize = computeBlockSize(fromDim = actualAxis + 1)
                var outOffset = 0
                repeat(blockCount) {
                    for (i in 0 until shape[actualAxis]) {
                        val zero = zeros?.get(i)?.toFloat() ?: 0f
                        for (j in 0 until blockSize) output[j + outOffset] = (this[j + outOffset].toFloat() - zero) * scale[i]
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

        return MutablePrimitiveNDArray(array.copyOfRange(start, start + rowLength), Strides(dims))
    }

    // TODO check if step == 1 and use Arrays.copy
    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, additionalOffset: Int) {
        array as LateInitPrimitiveArray
        for (index in range) {
            array.putNext(this.array[offset + additionalOffset + index])
        }
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArray {
        val newShape = IntArray(shape.size) {
            val length = abs(ends[it] - starts[it])
            val rest = length % abs(steps[it])
            (length / abs(steps[it])) + if (rest != 0) 1 else 0
        }

        val newStrides = Strides(newShape)
        val newArray = LateInitPrimitiveArray(newStrides.linearSize)

        slice(newArray, 0, 0, shape, starts, ends, steps)

        return MutablePrimitiveNDArray(newArray.getArray(), newStrides)
    }

    override fun min(): PrimitiveType {
        var min = PrimitiveType.MAX_VALUE
        for (index in 0 until linearSize) {
            val tmp = array[offset + index]
            if (tmp < min) min = tmp
        }

        return min
    }

    override fun max(): PrimitiveType {
        var max = PrimitiveType.MIN_VALUE
        for (index in 0 until linearSize) {
            val tmp = array[offset + index]
            if (tmp > max) max = tmp
        }

        return max
    }

    override fun sum(): PrimitiveType {
        var sum = (0).toPrimitive()
        for (index in 0 until linearSize) {
            sum = (sum + array[offset + index]).toPrimitive()
        }

        return sum
    }

    override fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArray {
        val output = MutablePrimitiveNDArray(PrimitiveArray(linearSize), strides)
        val actualAxis = indexAxis(axis)

        val blockSize = computeBlockSize(fromDim = actualAxis + 1)
        val batchSize = computeBlockSize(fromDim = actualAxis)
        val numBatches = computeBlockSize(toDim = actualAxis)
        val numBlocks = batchSize / blockSize
        repeat(numBatches) { batchIdx ->
            val dstOff = if (!reverse) batchIdx * batchSize else (numBatches - batchIdx) * batchSize - 1
            if (!exclusive) {
                if (!reverse)
                    this.array.copyInto(output.array, dstOff, dstOff, dstOff + blockSize)
                else
                    this.array.copyInto(output.array, dstOff - blockSize + 1, dstOff - blockSize + 1, dstOff + 1)
            }

            if (!reverse) {
                for (i in 1 until numBlocks) {
                    for (j in 0 until blockSize) {
                        val currentOff = dstOff + i * blockSize + j
                        val thisOff = if (!exclusive) currentOff else currentOff - blockSize
                        output.array[currentOff] = (output.array[currentOff - blockSize] + this.array[thisOff]).toPrimitive()
                    }
                }
            } else {
                for (i in 1 until numBlocks) {
                    for (j in blockSize - 1 downTo 0) {
                        val currentOff = dstOff - i * blockSize - j
                        val thisOff = if (!exclusive) currentOff else currentOff + blockSize
                        output.array[currentOff] = (output.array[currentOff + blockSize] + this.array[thisOff]).toPrimitive()
                    }
                }
            }
        }
        return output
    }

    override fun plus(other: NumberNDArray): MutableNumberNDArray = plus(other, MutablePrimitiveNDArray(PrimitiveArray(linearSize), strides))

    private fun plusScalar(array: PrimitiveArray, offset: Int, size: Int, scalar: PrimitiveType, destination: PrimitiveArray, destinationOffset: Int) {
        for (index in 0 until size) {
            destination[destinationOffset + index] = (array[offset + index] + scalar).toPrimitive()
        }
    }

    override fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array[destination.offset] = (this.array[this.offset] + other.array[other.offset]).toPrimitive()
            this.isScalar() -> plusScalar(other.array, other.offset, other.linearSize, this.array[this.offset], destination.array, destination.offset)
            other.isScalar() -> plusScalar(this.array, this.offset, this.linearSize, other.array[other.offset], destination.array, destination.offset)
            else -> this.applyWithBroadcast(other, destination, false) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] + right.array[right.offset + index]).toPrimitive()
                }
            }
        }

        return destination
    }

    override fun minus(other: NumberNDArray): MutableNumberNDArray = minus(other, MutablePrimitiveNDArray(PrimitiveArray(linearSize), strides))

    override fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array[destination.offset] = (this.array[this.offset] - other.array[other.offset]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[other.offset]
                for (index in 0 until this.linearSize) {
                    destination[destination.offset + index] = (this.array[this.offset + index] - scalar).toPrimitive()
                }
            }
            this.isScalar() -> error("Subtraction of a matrix from a scalar is prohibited")
            else -> this.applyWithBroadcast(other, destination, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] - right.array[right.offset + index]).toPrimitive()
                }
            }
        }

        return destination
    }

    override fun times(other: NumberNDArray): MutableNumberNDArray = times(other, MutablePrimitiveNDArray(PrimitiveArray(linearSize), strides))

    private fun timesScalar(array: PrimitiveArray, offset: Int, size: Int, scalar: PrimitiveType, destination: PrimitiveArray, destinationOffset: Int) {
        for (index in 0 until size) {
            destination[destinationOffset + index] = (array[offset + index] * scalar).toPrimitive()
        }
    }

    override fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array[destination.offset] = (this.array[this.offset] * other.array[other.offset]).toPrimitive()
            this.isScalar() -> timesScalar(other.array, other.offset, other.linearSize, this.array[this.offset], destination.array, destination.offset)
            other.isScalar() -> timesScalar(this.array, this.offset, this.linearSize, other.array[other.offset], destination.array, destination.offset)
            else -> this.applyWithBroadcast(other, destination, false) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] * right.array[right.offset + index]).toPrimitive()
                }
            }
        }

        return destination
    }

    override fun div(other: NumberNDArray): MutableNumberNDArray = div(other, MutablePrimitiveNDArray(PrimitiveArray(linearSize), strides))

    override fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array[destination.offset] = (this.array[this.offset] / other.array[other.offset]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[other.offset]
                for (index in 0 until this.linearSize) {
                    destination[destination.offset + index] = (this.array[this.offset + index] / scalar).toPrimitive()
                }
            }
            this.isScalar() -> error("Division of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, destination, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] / right.array[right.offset + index]).toPrimitive()
                }
            }
        }

        return destination
    }

    override fun dot(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        other as PrimitiveNDArray; destination as MutablePrimitiveNDArray
        require(shape.size == 2 && other.shape.size == 2)
        require(shape[1] == other.shape[0])

        val N = this.shape[0]
        val M = other.shape[1]
        val K = this.shape[1]

        for (n in 0 until N) {
            val dIdx = n * M + destination.offset
            val lIdx = n * K + this.offset
            for (k in 0 until K) {
                val temp = this.array[lIdx + k]
                val rIdx = k * M + other.offset
                for (m in 0 until M) {
                    destination.array[dIdx + m] = (destination.array[dIdx + m] + temp * other.array[rIdx + m]).toPrimitive()
                }
            }
        }

        return destination
    }

    override fun gemm(m: Int, n: Int, k: Int, alpha: Double, lda: Int, b: NDArray, ldb: Int, beta: Double, c: MutableNDArray, ldc: Int, aOffset: Int, bOffset: Int, cOffset: Int, transposeA: Boolean, transposeB: Boolean): MutableNDArray {
        b as PrimitiveNDArray; c as MutablePrimitiveNDArray
        val betaPrimitive = beta.toPrimitive()
        if (beta != 1.0) {
            for (i in 0 until m) {
                val cIdx = i * ldc + cOffset
                for (j in 0 until n) {
                    c[cIdx + j] = (betaPrimitive * c[cIdx + j]).toPrimitive()
                }
            }
        }

        val alphaPrimitive = alpha.toPrimitive()
        when {
            transposeA && transposeB -> {
                for (t in 0 until m) {
                    for (j in 0 until n) {
                        val cIdx = t * ldc + j + cOffset
                        for (i in 0 until k) {
                            val aIdx = i * lda + t + aOffset
                            val bIdx = j * ldb + i + bOffset
                            c[cIdx] = (alphaPrimitive * this[aIdx] * b[bIdx] + c[cIdx]).toPrimitive()
                        }
                    }
                }
            }
            transposeA -> {
                for (t in 0 until m) {
                    for (j in 0 until n) {
                        val cIdx = t * ldc + j + cOffset
                        for (i in 0 until k) {
                            val aIdx = i * lda + t + aOffset
                            val bIdx = i * ldb + j + bOffset
                            c[cIdx] = (alphaPrimitive * this[aIdx] * b[bIdx] + c[cIdx]).toPrimitive()
                        }
                    }
                }
            }
            transposeB -> {
                for (t in 0 until m) {
                    val aIdx = t * lda + aOffset
                    for (j in 0 until n) {
                        val cIdx = t * ldc + j + cOffset
                        val bIdx = j * ldb + bOffset
                        for (i in 0 until k) {
                            c[cIdx] = (alphaPrimitive * this[aIdx + i] * b[bIdx + i] + c[cIdx]).toPrimitive()
                        }
                    }
                }
            }
            else -> {
                for (t in 0 until m) {
                    val cIdx = t * ldc + cOffset
                    val aIdx = t * lda + aOffset
                    for (i in 0 until k) {
                        val temp = (alphaPrimitive * this[aIdx + i]).toPrimitive()
                        val bIdx = i * ldb + bOffset
                        for (j in 0 until n) {
                            c[cIdx + j] = (temp * b[bIdx + j] + c[cIdx + j]).toPrimitive()
                        }
                    }
                }
            }
        }

        return c
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutablePrimitiveNDArray(array.copyOf(), strides, offset)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PrimitiveNDArray) return false

        if (type != other.type) return false
        if (offset != other.offset) return false
        if (strides != other.strides) return false
        if (array != other.array) return false

        return true
    }

    override fun hashCode(): Int {
        var result = array.hashCode()
        result = 31 * result + strides.hashCode()
        result = 31 * result + offset
        result = 31 * result + type.hashCode()
        return result
    }
}

@PrimitiveClass
@ExperimentalUnsignedTypes
open class MutablePrimitiveNDArray(array: PrimitiveArray, strides: Strides = Strides.empty(), offset: Int = 0) : PrimitiveNDArray(array, strides, offset), MutableNumberNDArray {
    override fun set(index: Int, value: Any) {
        array[index] = value as PrimitiveType
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutablePrimitiveNDArray(array, strides)
    }

    override fun viewMutable(vararg axes: Int): MutableNumberNDArray {
        val (additionalOffset, newShape) = viewHelper(axes, strides)
        return MutablePrimitiveNDArray(array, Strides(newShape), offset + additionalOffset)
    }

    override fun fill(value: Any, from: Int, to: Int) {
        value as PrimitiveType
        array.fill(value, from, to)
    }

    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNumberNDArray {
        function as PrimitiveMap
        for (index in 0 until linearSize) {
            array[offset + index] = function.apply(array[offset + index])
        }

        return this
    }

    override fun erf(): MutableNumberNDArray {
        return this.mapMutable(object : PrimitiveMap {
            override fun apply(value: PrimitiveType): PrimitiveType = erfFor(value)
        })
    }

    override operator fun plusAssign(other: NDArray) {
        other as PrimitiveNDArray
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] + other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[other.offset]
                for (index in 0 until this.linearSize) {
                    this.array[this.offset + index] = (this.array[this.offset + index] + scalar).toPrimitive()
                }
            }
            this.isScalar() -> error("Plus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                // TODO change to real plusAssign
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] + right.array[right.offset + index]).toPrimitive()
                }
            }
        }
    }

    override operator fun minusAssign(other: NDArray) {
        other as PrimitiveNDArray
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] - other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[other.offset]
                for (index in 0 until this.linearSize) {
                    this.array[this.offset + index] = (this.array[this.offset + index] - scalar).toPrimitive()
                }
            }
            this.isScalar() -> error("Plus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] - right.array[right.offset + index]).toPrimitive()
                }
            }
        }
    }

    override operator fun timesAssign(other: NDArray) {
        other as PrimitiveNDArray
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] * other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[other.offset]
                for (index in 0 until this.linearSize) {
                    this.array[this.offset + index] = (this.array[this.offset + index] * scalar).toPrimitive()
                }
            }
            this.isScalar() -> error("Plus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] * right.array[right.offset + index]).toPrimitive()
                }
            }
        }
    }

    override operator fun divAssign(other: NDArray) {
        other as PrimitiveNDArray
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] / other.array[0]).toPrimitive()
            other.isScalar() -> {
                val scalar = other.array[other.offset]
                for (index in 0 until this.linearSize) {
                    this.array[this.offset + index] = (this.array[this.offset + index] / scalar).toPrimitive()
                }
            }
            this.isScalar() -> error("Plus assign of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, this, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (index in 0 until left.linearSize) {
                    dest.array[dest.offset + index] = (left.array[left.offset + index] / right.array[right.offset + index]).toPrimitive()
                }
            }
        }
    }

    override fun placeFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as PrimitiveNDArray
        other.array.copyInto(this.array, offset, startInOther, endInOther)
    }

    override fun placeAllFrom(offset: Int, other: NDArray) {
        other as PrimitiveNDArray
        other.array.copyInto(this.array, offset)
    }

    override fun reshape(strides: Strides): MutableNumberNDArray {
        this.strides = strides
        return this
    }

    // TODO separate from PrimitiveArray (maybe LateInitArray will help)
    private fun transposeRec(prevArray: PrimitiveArray, newArray: PrimitiveArray, prevStrides: Strides, newStrides: Strides, index: Int, prevOffset: Int, newOffset: Int, permutation: IntArray) {
        if (index != newStrides.shape.lastIndex) {
            val temp = prevStrides.strides[permutation[index]]
            val temp2 = newStrides.strides[index]
            for (i in 0 until newStrides.shape[index])
                transposeRec(prevArray, newArray, prevStrides, newStrides, index + 1, prevOffset + temp * i,
                    newOffset + temp2 * i, permutation)
        } else {
            val temp = prevStrides.strides[permutation[index]]
            if (temp == 1) {
                prevArray.copyInto(newArray, newOffset, prevOffset, prevOffset + newStrides.shape[index])
            } else {
                for (i in 0 until newStrides.shape[index]) {
                    newArray[newOffset + i] = prevArray[prevOffset + i * temp]
                }
            }
        }
    }

    override fun transpose(permutations: IntArray): MutableNumberNDArray {
        val newStrides = strides.transpose(permutations)
        transposeRec(array.copyOf(), array, strides, newStrides, 0, 0, 0, permutations)
        return this.reshape(newStrides)
    }

    override fun transpose2D(): MutableNDArray {
        require(rank == 2)

        val newShape = shape.reversedArray()
        val newStrides = Strides(newShape)

        val tmp = array.copyOf()
        for (j in (0 until shape[1])) {
            val ind = j * shape[0]
            for (i in (0 until shape[0])) {
                array[ind + i] = (tmp[i * shape[1] + j]).toPrimitive()
            }
        }

        return this.reshape(newStrides)
    }

    override fun clean() = array.fill((0).toPrimitive())
}

@PrimitiveClass
class LateInitPrimitiveArray(size: Int) : LateInitArray {
    private val array = PrimitiveArray(size)
    private var index = 0

    fun putNext(value: PrimitiveType) {
        array[index] = value
        index++
    }

    fun getArray(): PrimitiveArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}

@PrimitiveClass
interface PrimitiveMap : PrimitiveToPrimitiveFunction {
    fun apply(value: PrimitiveType): PrimitiveType
}
