package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.broadcastShape
import org.jetbrains.research.kotlin.inference.math.viewHelper
import org.jetbrains.research.kotlin.inference.onnx.TensorProto


fun <T> allocateHelper(shape: IntArray, otherShape: IntArray, type: TensorProto.DataType): MutableTypedNDArray<T> {
    return if (shape.contentEquals(otherShape))
        allocateNDArray(type, Strides(shape))
    else
        allocateNDArray(type, Strides(broadcastShape(shape, otherShape)))
}

fun <T> TypedNDArray<T>.view(vararg axes: Int): TypedNDArray<T> {
    val (additionalOffset, newShape) = viewHelper(axes, strides)
    return createNDArray(type, array, Strides(newShape), offset + additionalOffset)
}


fun <T> MutableTypedNDArray<T>.viewMutable(vararg axes: Int): MutableTypedNDArray<T> {
    val (additionalOffset, newShape) = viewHelper(axes, strides)
    return createMutableNDArray(type, array, Strides(newShape), offset + additionalOffset)
}
