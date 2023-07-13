package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.toIntArray

sealed class Slice(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in SliceVer10.VERSION.asRange() -> SliceVer10(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}


class SliceVer10(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Slice(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val DATA_TYPE_CONSTRAINTS = ALL_DATA_TYPES
        private val INDEX_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT64, TensorProto.DataType.INT32)

        private val INPUTS_INFO = listOf(
            IOInfo(0, DATA_TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, INDEX_TYPE_CONSTRAINTS, "starts", optional = false, differentiable = false),
            IOInfo(2, INDEX_TYPE_CONSTRAINTS, "ends", optional = false, differentiable = false),
            IOInfo(3, INDEX_TYPE_CONSTRAINTS, "axes", optional = true, differentiable = false),
            IOInfo(4, INDEX_TYPE_CONSTRAINTS, "steps", optional = true, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, DATA_TYPE_CONSTRAINTS, "output", optional = false, differentiable = true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("Slice", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val data = inputs[0]!!
        val shape = data.data.shape

        val incompleteAxes = inputs.getOrNull(3)?.let {
            if (it.data.type == DataType.LONG) {
                val pointer = (it.data as LongNDArray).array.pointer()
                IntArray(it.data.linearSize) {
                    val axis = pointer.getAndIncrement().toInt()
                    if (axis < 0) shape.size + axis else axis
                }
            } else {
                val pointer = (it.data as IntNDArray).array.pointer()
                IntArray(it.data.linearSize) {
                    val axis = pointer.getAndIncrement()
                    if (axis < 0) shape.size + axis else axis
                }
            }

        } ?: shape.indices.toIntArray()

        val incompleteSteps = inputs.getOrNull(4)?.let {
            require(it.data.linearSize == incompleteAxes.size) { "Input 'steps' must be same size as 'axes'" }
            if (it.data.type == DataType.LONG) {
                val pointer = (it.data as LongNDArray).array.pointer()
                IntArray(incompleteAxes.size) {
                    val step = pointer.getAndIncrement().toInt()
                    require(step != 0) { "Input 'steps' must not contains zeros " }
                    step
                }
            } else {
                val pointer = (it.data as IntNDArray).array.pointer()
                IntArray(incompleteAxes.size) {
                    val step = pointer.getAndIncrement()
                    require(step != 0) { "Input 'steps' must not contains zeros " }
                    step
                }
            }
        } ?: IntArray(shape.size) { 1 }

        val incompleteStarts = inputs[1]!!.data.let {
            require(it.linearSize == incompleteAxes.size) { "Input 'starts' must be same size as 'axes'" }
            if (it.type == DataType.LONG) {
                val pointer = (it as LongNDArray).array.pointer()
                IntArray(incompleteAxes.size) { index ->
                    var start = pointer.getAndIncrement()
                    val dim = shape[incompleteAxes[index]].toLong()
                    start = if (start < 0) dim + start else start
                    start = if (start >= dim) (if (incompleteSteps[index] > 0) dim else dim - 1) else start
                    if (start < 0) 0 else start.toInt()
                }
            } else {
                val pointer = (it as IntNDArray).array.pointer()
                IntArray(incompleteAxes.size) { index ->
                    var start = pointer.getAndIncrement().toLong()
                    val dim = shape[incompleteAxes[index]].toLong()
                    start = if (start < 0) dim + start else start
                    start = if (start >= dim) (if (incompleteSteps[index] > 0) dim else dim - 1) else start
                    if (start < 0) 0 else start.toInt()
                }
            }
        }

        val incompleteEnds = inputs[2]!!.data.let {
            require(it.linearSize == incompleteAxes.size) { "Input 'ends' must be same size as 'axes'" }
            if (it.type == DataType.LONG) {
                val pointer = (it as LongNDArray).array.pointer()
                IntArray(incompleteAxes.size) { index ->
                    var end = pointer.getAndIncrement()
                    val dim = shape[incompleteAxes[index]].toLong()
                    end = if (end < 0) dim + end else end
                    end = if (end > dim) dim else end
                    if (end < 0) (if (incompleteSteps[index] > 0) 0 else -1) else end.toInt()
                }
            } else {
                val pointer = (it as IntNDArray).array.pointer()
                IntArray(incompleteAxes.size) { index ->
                    var end = pointer.getAndIncrement().toLong()
                    val dim = shape[incompleteAxes[index]].toLong()
                    end = if (end < 0) dim + end else end
                    end = if (end > dim) dim else end
                    if (end < 0) (if (incompleteSteps[index] > 0) 0 else -1) else end.toInt()
                }
            }
        }

        val starts = IntArray(shape.size)
        val ends = IntArray(shape.size)
        val steps = IntArray(shape.size)

        for (axis in shape.indices) {
            val index = incompleteAxes.indexOf(axis)
            if (index == -1) {
                starts[axis] = 0
                ends[axis] = shape[axis]
                steps[axis] = 1
            } else {
                starts[axis] = incompleteStarts[index]
                ends[axis] = incompleteEnds[index]
                steps[axis] = incompleteSteps[index]
            }
        }

        return listOf((data.data.slice(starts, ends, steps) as NDArrayCore).asTensor("output"))
    }
}
