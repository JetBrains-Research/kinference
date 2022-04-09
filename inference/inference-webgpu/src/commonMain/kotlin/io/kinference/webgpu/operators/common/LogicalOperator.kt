package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.NDArrayInfo
import io.kinference.webgpu.ndarray.WebGPUDataType

abstract class LogicalOperator(
    info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : CachingShaderOperator(info, attributes, inputs, outputs) {
    abstract fun operation(input0: String, input1: String, output: String): String

    override fun operatorImplementation(inputInfo: List<NDArrayInfo?>, context: WebGPUContext): Operator<WebGPUTensor, WebGPUTensor> =
        when {
            inputInfo[0]!!.shape.contentEquals(inputInfo[1]!!.shape) -> ArithmeticOperatorWithoutBroadcast(
                context.gpuState.device, this::operation, inputInfo[0]!!, WebGPUDataType.INT32,
                info = info, attributes = attributes, inputs = inputs, outputs = outputs
            )
            else -> ArithmeticOperatorWithBroadcast(
                context.gpuState.device, this::operation, inputInfo[0]!!, inputInfo[1]!!, WebGPUDataType.INT32,
                info = info, attributes = attributes, inputs = inputs, outputs = outputs
            )
        }
}
