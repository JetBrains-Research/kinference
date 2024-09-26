package io.kinference.ndarray.arrays

import io.kinference.ndarray.arrays.pointers.*
import io.kinference.ndarray.arrays.tiled.BooleanTiledArray
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.ndarray.blockSizeByStrides
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsBoolean
import io.kinference.ndarray.extensions.isTransposeReshape
import io.kinference.primitives.types.DataType
import io.kinference.utils.inlines.InlineBoolean
import io.kinference.utils.inlines.InlineInt
import kotlin.jvm.JvmName
import kotlin.math.abs

interface BooleanMap : PrimitiveToPrimitiveFunction {
    fun apply(value: Boolean): Boolean
}

fun interface BooleanBinaryOperation {
    operator fun invoke(first: Boolean, second: Boolean): Boolean
}

open class BooleanNDArray(var array: BooleanTiledArray, strides: Strides) : NDArrayCore {
    override val type: DataType = DataType.BOOLEAN

    final override var strides: Strides = strides
        protected set

    internal val blocksInRow: Int
        get() = when {
            strides.linearSize == 0 -> 0
            strides.shape.isEmpty() -> 1
            else -> strides.shape.last() / array.blockSize
        }

    override suspend fun clone(): NDArrayCore {
        return BooleanNDArray(array.copyOf(), Strides(shape))
    }

    override suspend fun close() = Unit

    override fun view(vararg axes: Int): BooleanNDArray {
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

    override fun get(index: IntArray): Boolean {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        return array[linearIndex]
    }

    override fun getLinear(index: Int): Boolean {
        return array[index]
    }

    override fun singleValue(): Boolean {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array.blocks[0][0]
    }

    override suspend fun toMutable(): MutableBooleanNDArray {
        return MutableBooleanNDArray(array.copyOf(), strides)
    }

    override suspend fun copyIfNotMutable(): MutableBooleanNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override suspend fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableBooleanNDArray {
        function as BooleanMap
        destination as MutableBooleanNDArray
        for (index in 0 until destination.linearSize) {
            destination.array[index] = function.apply(this.array[index])
        }

        return destination
    }

    override suspend fun map(function: PrimitiveToPrimitiveFunction): MutableNDArrayCore = map(function, MutableBooleanNDArray(strides))

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableBooleanNDArray {
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

    suspend fun or(other: BooleanNDArray, destination: MutableBooleanNDArray): BooleanNDArray {
        return broadcastTwoTensorsBoolean(this, other, destination) {
                left: Boolean, right: Boolean -> left or right
        }
    }

    suspend infix fun or(other: BooleanNDArray) = or(other, MutableBooleanNDArray(broadcastShape(listOf(this.shape, other.shape))))

    suspend fun and(other: BooleanNDArray, destination: MutableBooleanNDArray): BooleanNDArray {
        return broadcastTwoTensorsBoolean(this, other, destination) {
            left: Boolean, right: Boolean -> left and right
        }
    }

    suspend infix fun and(other: BooleanNDArray) = and(other, MutableBooleanNDArray(broadcastShape(listOf(this.shape, other.shape))))

    suspend fun xor(other: BooleanNDArray, destination: MutableBooleanNDArray): BooleanNDArray {
        return broadcastTwoTensorsBoolean(this, other, destination) {
            left: Boolean, right: Boolean -> left xor right
        }
    }

    suspend infix fun xor(other: BooleanNDArray) = xor(other, MutableBooleanNDArray(broadcastShape(listOf(this.shape, other.shape))))

    override suspend fun concat(others: List<NDArray>, axis: Int): MutableBooleanNDArray {
        val actualAxis = indexAxis(axis)

        val inputs = others.toMutableList().also { it.add(0, this) }
        val resultShape = shape.copyOf()
        resultShape[actualAxis] = inputs.sumOf { it.shape[actualAxis] }

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

    override suspend fun expand(shape: IntArray): MutableBooleanNDArray {
        val outputShape = broadcastShape(listOf(this.shape, shape))
        val output = MutableBooleanNDArray(outputShape)
        Broadcasting.applyWithBroadcast(listOf(this), output) { inputs: List<NDArray>, destination: MutableNDArray ->
            destination as MutableBooleanNDArray
            val input = inputs[0] as BooleanNDArray
            destination.copyFrom(0, input)
        }

        return output
    }

    override suspend fun nonZero(): LongNDArray {
        if (isScalar()) {
            val value = singleValue()
            return if (value)
                LongNDArray(LongTiledArray(emptyArray()), Strides(intArrayOf(1, 0)))
            else
                LongNDArray(Strides(intArrayOf(1, 1))) { _: InlineInt -> 0L }
        }
        val ndIndexSize = shape.size
        var totalElements = 0
        val inputPointer = array.pointer()
        val indicesArray = IntArray(linearSize * ndIndexSize)
        this.ndIndices { ndIndex ->
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
                resultPointer.set(indicesArray[j * ndIndexSize + i].toLong())
                resultPointer.increment()
            }
        return LongNDArray(indicesByDim, nonZeroStrides)
    }

    override suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): BooleanNDArray {
        TODO("Not yet implemented")
    }

    override suspend fun tile(repeats: IntArray): BooleanNDArray {
        TODO("Not yet implemented")
    }

    private fun transposeByBlocks(permutations: IntArray): BooleanNDArray {
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

    override suspend fun transpose(permutations: IntArray): BooleanNDArray {
        require(permutations.size == rank)
        require(permutations.all { it in permutations.indices })

        val outputStrides = strides.transpose(permutations)

        if (isTransposeReshape(permutations)) {
            return reshape(outputStrides)
        }

        if (permutations.lastIndex == permutations.last()) {
            return transposeByBlocks(permutations)
        }

        val outputArray = MutableBooleanNDArray(outputStrides)

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

    override suspend fun transpose2D(): BooleanNDArray {
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

    override suspend fun reshape(strides: Strides): BooleanNDArray {
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
        suspend fun scalar(value: Boolean): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(1, 1) { value }, Strides.EMPTY)
        }

        suspend fun eyeLike(shape: IntArray, k: Int = 0): BooleanNDArray {
            require(shape.size == 2) { "EyeLike is only supported for tensors of rank=2, current shape rank: ${shape.size}" }

            return BooleanNDArray(shape) { it: IntArray ->
                val (row, column) = it
                (column - k) == row
            }
        }

        @JvmName("invokeStrides")
        suspend operator fun invoke(strides: Strides): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(strides), strides)
        }

        @JvmName("invokeStridesInlineInt")
        suspend operator fun invoke(strides: Strides, init: (InlineInt) -> Boolean): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(strides, init), strides)
        }

        @JvmName("invokeStridesIntArray")
        suspend operator fun invoke(strides: Strides, init: (IntArray) -> Boolean): BooleanNDArray {
            val iterator = NDIndexer(strides)
            return BooleanNDArray(strides) { _: InlineInt -> init(iterator.next()) }
        }

        @JvmName("invokeStridesTiled")
        suspend operator fun invoke(array: BooleanTiledArray, strides: Strides): BooleanNDArray {
            val blockSize = blockSizeByStrides(strides)
            return if (blockSize == array.blockSize) {
                BooleanNDArray(array, strides)
            }
            else {
                val pointer = BooleanPointer(array)
                BooleanNDArray(strides) { _: InlineInt -> pointer.getAndIncrement() }
            }
        }

        @JvmName("invokeShape")
        suspend operator fun invoke(shape: IntArray): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(shape), Strides(shape))
        }

        @JvmName("invokeShapeVarArg")
        suspend operator fun invoke(vararg shape: Int): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(shape), Strides(shape))
        }

        @JvmName("invokeShapeInlineInt")
        suspend operator fun invoke(shape: IntArray, init: (InlineInt) -> Boolean): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(shape, init), Strides(shape))
        }

        @JvmName("invokeShapeVarArgInlineInt")
        suspend operator fun invoke(vararg shape: Int, init: (InlineInt) -> Boolean): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(shape, init), Strides(shape))
        }

        @JvmName("invokeShapeIntArray")
        suspend operator fun invoke(shape: IntArray, init: (IntArray) -> Boolean): BooleanNDArray {
            return invoke(Strides(shape), init)
        }

        @JvmName("invokeShapeVarArgIntArray")
        suspend operator fun invoke(vararg shape: Int, init: (IntArray) -> Boolean): BooleanNDArray {
            return invoke(shape, init)
        }
    }
}

class MutableBooleanNDArray(array: BooleanTiledArray, strides: Strides = Strides.EMPTY): BooleanNDArray(array, strides), MutableNDArrayCore {
    override fun set(index: IntArray, value: Any) {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        array[linearIndex] = value as Boolean
    }

    override fun setLinear(index: Int, value: Any) {
        array[index] = value as Boolean
    }

    override fun viewMutable(vararg axes: Int): MutableBooleanNDArray {
        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }
        val offsetBlocks = offset / array.blockSize

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = BooleanTiledArray(copyBlocks)

        return MutableBooleanNDArray(newArray, newStrides)
    }

    override suspend fun copyIfNotMutable(): MutableBooleanNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override suspend fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableBooleanNDArray {
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

    suspend fun not(): MutableBooleanNDArray {
        return mapMutable(object : BooleanMap {
            override fun apply(value: Boolean): Boolean = value.not()
        })
    }

    companion object {
        suspend fun scalar(value: Boolean): MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(1, 1) { value }, Strides.EMPTY)
        }

        @JvmName("invokeStrides")
        suspend operator fun invoke(strides: Strides) : MutableBooleanNDArray{
            return MutableBooleanNDArray(BooleanTiledArray(strides), strides)
        }

        @JvmName("invokeStridesInlineInt")
        suspend operator fun invoke(strides: Strides, init: (InlineInt) -> Boolean) : MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(strides, init), strides)
        }

        @JvmName("invokeStridesIntArray")
        suspend operator fun invoke(strides: Strides, init: (IntArray) -> Boolean): MutableBooleanNDArray {
            val iterator = NDIndexer(strides)
            return MutableBooleanNDArray(strides) { _: InlineInt -> init(iterator.next()) }
        }

        @JvmName("invokeShape")
        suspend operator fun invoke(shape: IntArray) : MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(shape), Strides(shape))
        }

        @JvmName("invokeShapeVarArg")
        suspend operator fun invoke(vararg shape: Int): MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(shape), Strides(shape))
        }

        @JvmName("invokeShapeInlineInt")
        suspend operator fun invoke(shape: IntArray, init: (InlineInt) -> Boolean) : MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(shape, init), Strides(shape))
        }

        @JvmName("invokeShapeVarArgInlineInt")
        suspend operator fun invoke(vararg shape: Int, init: (InlineInt) -> Boolean): MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(shape, init), Strides(shape))
        }

        @JvmName("invokeShapeIntArray")
        suspend operator fun invoke(shape: IntArray, init: (IntArray) -> Boolean): MutableBooleanNDArray {
            return invoke(Strides(shape), init)
        }

        @JvmName("invokeShapeVarArgIntArray")
        suspend operator fun invoke(vararg shape: Int, init: (IntArray) -> Boolean): MutableBooleanNDArray {
            return invoke(shape, init)
        }
    }
}
