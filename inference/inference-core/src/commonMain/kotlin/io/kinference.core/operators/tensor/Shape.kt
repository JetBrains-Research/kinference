package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class Shape(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 15)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ShapeVer1.VERSION.asRange() -> ShapeVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

@ExperimentalTime
class ShapeVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Shape(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 15)
        private val INFO = OperatorInfo("Shape", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val tensor = inputs.first()!!
        val shape = tensor.data.shape

        val outputTensorShape = intArrayOf(shape.size)
        val data = LongTiledArray(outputTensorShape) { shape[it].toLong() }
        return listOf(createNDArray(DataType.LONG, data, outputTensorShape).asTensor("shape"))
    }
}
