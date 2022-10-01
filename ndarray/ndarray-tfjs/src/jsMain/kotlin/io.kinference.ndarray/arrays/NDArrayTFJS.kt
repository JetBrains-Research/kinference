package io.kinference.ndarray.arrays

import io.kinference.ndarray.activateDefaultBackend
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.resolveTFJSDataType
import io.kinference.primitives.types.DataType

abstract class NDArrayTFJS(tfjsArray: ArrayTFJS) : NDArray {
    init {
        if (!isActivated) {
            activateDefaultBackend()
            isActivated = true
        }
    }

    var tfjsArray = tfjsArray
        protected set

    override val strides
        get() = Strides(tfjsArray.shape.toIntArray())

    override val type: DataType = tfjsArray.dtype.resolveTFJSDataType()

    override fun close() {
        tfjsArray.dispose()
    }

    override fun gather(indices: NDArray, axis: Int, batchDims: Int): NDArrayTFJS {
        indices as NDArrayTFJS
        return tfjsArray.gather(indices.tfjsArray, axis, batchDims).toNDArray()
    }

    override fun squeeze(vararg axes: Int): NDArrayTFJS {
        return tfjsArray.squeeze(axes.toTypedArray()).toNDArray()
    }

    override fun unsqueeze(vararg axes: Int): NDArrayTFJS {
        fun indexAxisForUnsqueeze(axis: Int, shapeSize: Int): Int = if (axis < 0) shapeSize + axis else axis

        val actualAxes = axes.map { indexAxisForUnsqueeze(it, rank + axes.size) }.sorted()
        val newShape = shape.toMutableList()
        for (axis in actualAxes) {
            newShape.add(axis, 1)
        }

        return this.reshape(newShape.toIntArray())
    }

    override fun reshape(strides: Strides): NDArrayTFJS {
        return reshape(strides.shape)
    }

    override fun reshape(shape: IntArray): NDArrayTFJS {
        return tfjsArray.reshape(shape.toTypedArray()).toNDArray()
    }

    @Suppress("UNCHECKED_CAST")
    override fun stack(others: List<NDArray>, axis: Int): NDArrayTFJS {
        others as List<NDArrayTFJS>
        return tfjsArray.stack(*others.getArrays(), axis = axis).toNDArray()
    }

    @Suppress("UNCHECKED_CAST")
    override fun concat(others: List<NDArray>, axis: Int): NDArrayTFJS {
        others as List<NDArrayTFJS>
        return tfjsArray.concat(*others.getArrays(), axis = axis).toNDArray()
    }

    override fun tile(repeats: IntArray): NDArrayTFJS {
        return tfjsArray.tile(repeats.toTypedArray()).toNDArray()
    }

    override fun transpose(permutations: IntArray): NDArrayTFJS {
        return tfjsArray.transpose(permutations.toTypedArray()).toNDArray()
    }

    override fun transpose2D(): NDArrayTFJS = transpose(intArrayOf(1, 0))

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): NDArrayTFJS {
        val result = tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray())
        return result.toNDArray()
    }

    override fun split(parts: Int, axis: Int): List<NDArrayTFJS> {
        return tfjsArray.split(parts, axis).map { it.toNDArray() }
    }

    override fun split(split: IntArray, axis: Int): List<NDArrayTFJS> {
        return tfjsArray.split(split.toTypedArray(), axis).map { it.toNDArray() }
    }

    companion object {
        private var isActivated = false

        private fun Array<Int>.times() = this.fold(1, Int::times)

        internal fun zero(dtype: String): NDArrayTFJS {
            val zero = when (dtype) {
                "int32" -> intScalar(0)
                "float32" -> floatScalar(0f)
                "bool" -> booleanScalar(false)
                else -> error("Unsupported data type: $dtype")
            }
            return zero
        }

        fun float(values: FloatArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun int(values: IntArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "int32"))
        fun boolean(values: Array<Boolean>, shape: Array<Int>) = BooleanNDArrayTFJS(tensor(values, shape))

        fun float(shape: Array<Int>, init: (Int) -> Float) = NumberNDArrayTFJS(tensor(FloatArray(shape.times(), init), shape, "float32"))
        fun int(shape: Array<Int>, init: (Int) -> Int) = NumberNDArrayTFJS(tensor(IntArray(shape.times(), init), shape, "int32"))
        fun boolean(shape: Array<Int>, init: (Int) -> Boolean) = BooleanNDArrayTFJS(tensor(Array(shape.times()) { init(it) }, shape))

        fun float(shape: Array<Int>, init: (IntArray) -> Float): NumberNDArrayTFJS {
            val ndIterator = NDIndexer(shape.toIntArray())
            val array = FloatArray(shape.times()) { init(ndIterator.next()) }
            return NumberNDArrayTFJS(tensor(array, shape, "float32"))
        }

        fun int(shape: Array<Int>, init: (IntArray) -> Int): NumberNDArrayTFJS {
            val ndIterator = NDIndexer(shape.toIntArray())
            val array = IntArray(shape.times()) { init(ndIterator.next()) }
            return NumberNDArrayTFJS(tensor(array, shape, "int32"))
        }

        fun boolean(shape: Array<Int>, init: (IntArray) -> Boolean): BooleanNDArrayTFJS {
            val ndIterator = NDIndexer(shape.toIntArray())
            val array = Array(shape.times()) { init(ndIterator.next()) }
            return BooleanNDArrayTFJS(tensor(array, shape))
        }

        fun floatScalar(value: Float) = NumberNDArrayTFJS(scalar(value, "float32"))
        fun intScalar(value: Int) = NumberNDArrayTFJS(scalar(value, "int32"))
        fun booleanScalar(value: Boolean) = BooleanNDArrayTFJS(scalar(value))

        fun floatZeros(shape: Array<Int>) = NumberNDArrayTFJS(tensor(FloatArray(shape.times()), shape, "float32"))
        fun intZeros(shape: Array<Int>) = NumberNDArrayTFJS(tensor(IntArray(shape.times()), shape, "int32"))
        fun booleanZeros(shape: Array<Int>) = BooleanNDArrayTFJS(tensor(Array(shape.times()) { false }, shape))

        fun floatOnes(shape: Array<Int>) = NumberNDArrayTFJS(tensor(FloatArray(shape.times()) { 1f }, shape, "float32"))
        fun intOnes(shape: Array<Int>) = NumberNDArrayTFJS(tensor(IntArray(shape.times()) { 1 }, shape, "int32"))
        fun booleanOnes(shape: Array<Int>) = BooleanNDArrayTFJS(tensor(Array(shape.times()) { true }, shape))

        fun floatRange(start: Float, stop: Float, step: Float) = NumberNDArrayTFJS(range(start, stop, step, "float32"))
        fun intRange(start: Int, stop: Int, step: Int) = NumberNDArrayTFJS(range(start, stop, step, "int32"))

        fun floatFill(shape: Array<Int>, value: Float) = NumberNDArrayTFJS(fill(shape, value, "float32"))
        fun intFill(shape: Array<Int>, value: Int) = NumberNDArrayTFJS(fill(shape, value, "int32"))
    }
}
