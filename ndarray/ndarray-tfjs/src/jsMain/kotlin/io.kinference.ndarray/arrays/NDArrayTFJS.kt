package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.core.*
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType

abstract class NDArrayTFJS(tfjsArray: ArrayTFJS) : NDArray {
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

    override fun stack(others: List<NDArray>, axis: Int): NDArrayTFJS {
        others as List<NDArrayTFJS>
        return tfjsArray.stack(*others.getArrays(), axis = axis).toNDArray()
    }
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
}

val NDArrayTFJS.dtype: String
    get() = tfjsArray.dtype

val NDArrayTFJS.shapeArray: Array<Int>
    get() = tfjsArray.shape
