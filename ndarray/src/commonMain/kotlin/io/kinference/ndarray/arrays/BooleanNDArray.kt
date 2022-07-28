package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.arrays.tiled.BooleanTiledArray
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.extensions.applyWithBroadcast
import io.kinference.ndarray.extensions.isScalar
import io.kinference.ndarray.extensions.ndIndexed
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType
import kotlin.math.abs
import kotlin.ranges.reversed

interface BooleanMap : PrimitiveToPrimitiveFunction {
    fun apply(value: Boolean): Boolean
}

open class BooleanNDArray(var array: BooleanTiledArray, strides: Strides) : NDArray {
    constructor(shape: IntArray) : this(BooleanTiledArray(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> Boolean) : this(BooleanTiledArray(shape, init), Strides(shape))

    constructor(strides: Strides) : this(BooleanTiledArray(strides), strides)
    constructor(strides: Strides, init: (Int) -> Boolean) : this(BooleanTiledArray(strides, init), strides)

    override val type: DataType = DataType.BOOLEAN

    final override var strides: Strides = strides
        protected set

    protected val blocksInRow: Int
        get() = when {
            strides.linearSize == 0 -> 0
            strides.shape.isEmpty() -> 1
            else -> strides.shape.last() / array.blockSize
        }

    override fun view(vararg axes: Int): NDArray {
        for ((i, axis) in axes.withIndex()) {
            require(shape[i] > axis)
        }

        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        if (array.blockSize == 0)
            return BooleanNDArray(array, newStrides)


        val offsetBlocks = offset / array.blockSize

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = BooleanTiledArray(copyBlocks)

        return BooleanNDArray(newArray, newStrides)
    }

    override fun singleValue(): Boolean {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array.blocks[0][0]
    }

    override fun reshapeView(newShape: IntArray): NDArray {
        return BooleanNDArray(array, Strides(newShape))
    }

    override fun toMutable(newStrides: Strides): MutableNDArray {
        return MutableBooleanNDArray(array.copyOf(), strides)
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNDArray {
        function as BooleanMap
        destination as MutableBooleanNDArray
        for (index in 0 until destination.linearSize) {
            destination.array[index] = function.apply(this.array[index])
        }

        return destination
    }

    override fun row(row: Int): MutableNDArray {
        val rowLength: Int = linearSize / shape[0]
        val start = row * rowLength
        val dims = shape.copyOfRange(1, rank)

        return MutableBooleanNDArray(BooleanTiledArray(Strides(dims)) { array[start + it] }, Strides(dims))
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArray {
        val newShape = IntArray(shape.size) {
            val length = abs(ends[it] - starts[it])
            val rest = length % abs(steps[it])
            (length / abs(steps[it])) + if (rest != 0) 1 else 0
        }

        val newStrides = Strides(newShape)
        val newArray = BooleanTiledArray(newStrides)

        if (newArray.size > 0) {
            slice(newArray.pointer(), this.array.pointer(), 0, 0, shape, starts, ends, steps)
        }

        return MutableBooleanNDArray(newArray, newStrides)
    }

    private fun slice(dst: BooleanPointer, src: BooleanPointer, offset: Int, axis: Int, shape: IntArray, starts: IntArray, ends: IntArray, steps: IntArray) {
        val start = starts[axis]
        val end = ends[axis]
        val step = steps[axis]

        val range = if (step > 0) (start until end step step) else (start downTo end + 1 step -step)

        if (axis == shape.size - 1) {
            for (index in range) {
                src.linearIndex = offset + index
                dst.set(src.get())
                dst.increment()
            }
        } else {
            var dim = 1
            for (ind in (axis + 1) until shape.size) dim *= shape[ind]

            for (index in range) {
                slice(dst, src, offset + index * dim, axis + 1, shape, starts, ends, steps)
            }
        }
    }

    private fun orScalar(array: BooleanTiledArray, scalar: Boolean, destination: BooleanTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        val arrayPointer = array.pointer()
        val destPointer = destination.pointer()

        arrayPointer.mapTo(destPointer, destination.size) { it || scalar }
    }

    fun or(other: BooleanNDArray, destination: MutableBooleanNDArray): BooleanNDArray {
        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = this.array.blocks[0][0] or other.array.blocks[0][0]
            this.isScalar() -> orScalar(other.array, this.array.blocks[0][0], destination.array)
            other.isScalar() -> orScalar(this.array, other.array.blocks[0][0], destination.array)
            else -> this.applyWithBroadcast(other, destination) { left, right, dest ->
                left as BooleanNDArray; right as BooleanNDArray; dest as MutableBooleanNDArray

                val leftPointer = left.array.pointer()
                val rightPointer = right.array.pointer()
                val destPointer = dest.array.pointer()

                destPointer.acceptDouble(leftPointer, rightPointer, dest.array.size) { _, a, b -> a || b }
            }
        }

        return destination
    }

    infix fun or(other: BooleanNDArray) = or(other, MutableBooleanNDArray(Broadcasting.broadcastShape(listOf(this.shape, other.shape))))

    override fun concatenate(others: List<NDArray>, axis: Int): MutableNDArray {
        val actualAxis = indexAxis(axis)

        val inputs = others.toMutableList().also { it.add(0, this) }
        val resultShape = shape.copyOf()
        resultShape[actualAxis] = inputs.sumBy { it.shape[actualAxis] }

        val result = MutableBooleanNDArray(resultShape)
        val resultPointer = result.array.pointer()

        val numIterations = resultShape.take(actualAxis).fold(1, Int::times)
        val pointersToSteps = inputs.map {
            require(it is BooleanNDArray)
            it.array.pointer() to it.shape.drop(actualAxis).fold(1, Int::times)
        }

        repeat(numIterations) {
            pointersToSteps.forEach { (pointer, numSteps) ->
                resultPointer.accept(pointer, numSteps) { _, src -> src }
            }
        }
        return result
    }

    override fun expand(shape: IntArray): MutableNDArray {
        val outputShape = Broadcasting.broadcastShape(listOf(this.shape, shape))
        val output = allocateNDArray(type, Strides(outputShape))
        Broadcasting.applyWithBroadcast(listOf(this), output) { inputs: List<NDArray>, destination: MutableNDArray ->
            destination as MutableBooleanNDArray
            val input = inputs[0] as BooleanNDArray
            destination.copyFrom(0, input)
        }

        return output
    }

    override fun nonZero(): LongNDArray {
        if (isScalar()) {
            val value = singleValue()
            return if (value)
                LongNDArray(LongTiledArray(emptyArray()), Strides(intArrayOf(1, 0)))
            else
                LongNDArray(Strides(intArrayOf(1, 1))) { 0L }
        }
        val ndIndexSize = shape.size
        var totalElements = 0
        val inputPointer = array.pointer()
        val indicesArray = LongArray(linearSize * ndIndexSize)
        this.ndIndexed { ndIndex ->
            if (inputPointer.getAndIncrement()) {
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

    override fun pad(pads: Array<Pair<Int, Int>>, mode: String, constantValue: NDArray?): NDArray {
        TODO("Not yet implemented")
    }

    override fun tile(repeats: IntArray): NDArray {
        TODO("Not yet implemented")
    }

    private fun transposeByBlocks(permutations: IntArray): NDArray {
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

        return BooleanNDArray(BooleanTiledArray(outputBlocks), outputStrides)
    }

    override fun transpose(permutations: IntArray): NDArray {
        require(permutations.size == rank)
        require(permutations.all { it in permutations.indices })

        val outputStrides = strides.transpose(permutations)

        if (isTransposeReshape(permutations)) {
            return reshape(outputStrides)
        }

        if (permutations.lastIndex == permutations.last()) {
            return transposeByBlocks(permutations)
        }

        val outputArray = allocateNDArray(type, outputStrides) as BooleanNDArray

        fun transposeRec(axis: Int, inputOffset: Int, outputOffset: Int) {
            when(axis) {
                shape.lastIndex -> {
                    val dims = outputStrides.shape[axis]
                    val inputStride = this.strides.strides[permutations[axis]]
                    var index = 0
                    outputArray.array.pointer(outputOffset).map(dims) { _: Boolean -> this.array[inputOffset + inputStride * index++] }
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

    override fun transpose2D(): NDArray {
        require(rank == 2)

        val outputShape = shape.reversedArray()
        val outputStrides = Strides(outputShape)
        val outputArray = BooleanTiledArray(outputStrides)

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

        return BooleanNDArray(outputArray, outputStrides)
    }

    override fun reshape(strides: Strides): BooleanNDArray {
        require(strides.linearSize == this.strides.linearSize) { "Linear size must be equal" }

        if (strides.shape.isNotEmpty() && this.shape.isNotEmpty() && strides.shape.last() != this.shape.last()) {
            val newArray = BooleanTiledArray(strides)
            this.array.copyInto(newArray)

            return BooleanNDArray(newArray, strides)
        }

        return BooleanNDArray(this.array, strides)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BooleanNDArray) return false

        if (type != other.type) return false
        if (strides != other.strides) return false
        if (array != other.array) return false

        return true
    }

    companion object {
        fun scalar(value: Boolean): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(1, 1) { value }, Strides.EMPTY)
        }

        operator fun invoke(array: BooleanTiledArray, strides: Strides): BooleanNDArray {
            val blockSize = blockSizeByStrides(strides)
            return if (blockSize == array.blockSize) {
                BooleanNDArray(array, strides)
            }
            else {
                val pointer = BooleanPointer(array)
                BooleanNDArray(strides) { pointer.getAndIncrement() }
            }
        }
    }
}

class MutableBooleanNDArray(array: BooleanTiledArray, strides: Strides = Strides.EMPTY): BooleanNDArray(array, strides), MutableNDArray {
    constructor(shape: IntArray) : this(BooleanTiledArray(shape), Strides(shape))

    override fun viewMutable(vararg axes: Int): MutableNDArray {
        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }
        val offsetBlocks = offset / array.blockSize

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = BooleanTiledArray(copyBlocks)

        return MutableBooleanNDArray(newArray, newStrides)
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        function as BooleanMap
        for (index in 0 until linearSize) {
            array[index] = function.apply(array[index])
        }

        return this
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as BooleanNDArray
        other.array.copyInto(this.array, offset, startInOther, endInOther)
    }

    override fun fill(value: Any, from: Int, to: Int) {
        array.fill(value as Boolean)
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        array as BooleanNDArray
        val (blockIndex, blockOffset) = array.array.indexFor(index)
        this.array.fill(array.array.blocks[blockIndex][blockOffset], from, to)
    }

    // TODO separate from PrimitiveArray (maybe LateInitArray will help)
    private fun transposeRec(prevArray: BooleanTiledArray, newArray: BooleanTiledArray, prevStrides: Strides, newStrides: Strides, index: Int, prevOffset: Int, newOffset: Int, permutation: IntArray) {
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
                var (newArrayBlock, newArrayOffset) = newArray.indexFor(newOffset)
                var (prevArrayBlock, prevArrayOffset) = prevArray.indexFor(prevOffset)

                val (deltaBlock, deltaOffset) = prevArray.indexFor(temp)

                var tempNewBlock = newArray.blocks[newArrayBlock]
                var tempPrevBlock = prevArray.blocks[prevArrayBlock]

                for (i in 0 until newStrides.shape[index]) {
                    tempNewBlock[newArrayOffset++] = tempPrevBlock[prevArrayOffset]

                    prevArrayBlock += deltaBlock
                    prevArrayOffset += deltaOffset

                    if (prevArrayOffset >= prevArray.blockSize) {
                        prevArrayBlock += prevArrayOffset / prevArray.blockSize
                        prevArrayOffset %= prevArray.blockSize
                    }

                    if (prevArrayBlock < prevArray.blocksNum)
                        tempPrevBlock = prevArray.blocks[prevArrayBlock]

                    if (newArrayOffset >= newArray.blockSize) {
                        newArrayOffset = 0

                        if (++newArrayBlock < newArray.blocksNum)
                            tempNewBlock = newArray.blocks[newArrayBlock]
                    }
                }
            }
        }
    }

    /*override fun transpose(permutations: IntArray): MutableNDArray {
        val newStrides = strides.transpose(permutations)
        val newArray = BooleanTiledArray(newStrides)
        array.copyInto(newArray)

        transposeRec(array, newArray, strides, newStrides, 0, 0, 0, permutations)

        this.strides = newStrides
        this.array = newArray
        return this
    }*/

    /*override fun transpose2D(): MutableNDArray {
        TODO("Not yet implemented")
    }*/

    override fun clean() {
        array.fill(false)
    }

    fun not(): MutableNDArray {
        return mapMutable(object : BooleanMap {
            override fun apply(value: Boolean): Boolean = value.not()
        })
    }

    companion object {
        fun scalar(value: Boolean): MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(1, 1) { value }, Strides.EMPTY)
        }
    }
}
