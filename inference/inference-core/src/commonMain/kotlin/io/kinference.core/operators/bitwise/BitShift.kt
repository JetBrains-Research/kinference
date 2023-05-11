package io.kinference.core.operators.bitwise

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.bitwise.shift.BitShiftDirection
import io.kinference.ndarray.extensions.bitwise.shift.bitShift
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto

sealed class BitShift(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in BitShiftVer11.VERSION.asRange() -> BitShiftVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of BitShift operator: $version")
            }
    }
}


class BitShiftVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    BitShift(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = UINT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("direction", setOf(AttributeProto.AttributeType.STRING), true)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false, differentiable = false),
            IOInfo(1, TYPE_CONSTRAINTS, "Y", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Z", optional = false, differentiable = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("BitShift", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val direction by attribute("direction") { it: String ->
        when (it) {
            "LEFT" -> BitShiftDirection.LEFT
            "RIGHT" -> BitShiftDirection.RIGHT
            else -> error("Attribute \"direction\" must be either \'LEFT\' or \'RIGHT\'")
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data
        val shiftTensor = inputs[1]!!.data

        require(input.type == shiftTensor.type)

        val output = when (input.type) {
            DataType.UINT -> (input as UIntNDArray).bitShift(shiftTensor as UIntNDArray, direction)
            DataType.USHORT -> (input as UShortNDArray).bitShift(shiftTensor as UShortNDArray, direction)
            DataType.UBYTE -> (input as UByteNDArray).bitShift(shiftTensor as UByteNDArray, direction)
            DataType.ULONG -> (input as ULongNDArray).bitShift(shiftTensor as ULongNDArray, direction)
            else -> error("")
        }

        return listOf(output.asTensor("Z"))
    }
}
