package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.ndarray.Strides
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.broadcasting.unsqueezeFirst
import io.kinference.operator.OperatorInfo
import io.kinference.webgpu.ndarray.ArrayInfo
import io.kinference.webgpu.utils.WORK_GROUP_SIZE_1D
import io.kinference.webgpu.utils.divUp

abstract class BroadcastingBinaryOperator(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : BinaryOperator(info, attributes, inputs, outputs) {
    abstract fun operation(input0: String, input1: String, output: String): String

    override fun outputInfo(inputInfo: List<ArrayInfo?>): List<ArrayInfo?> =
        listOf(ArrayInfo(Broadcasting.broadcastShape(listOf(inputInfo[0]!!.shape, inputInfo[1]!!.shape)), inputInfo[0]!!.type))

    private fun outputShape(inputInfo: List<ArrayInfo?>, outputInfo: List<ArrayInfo?>): IntArray = outputInfo[0]!!.let {
        if (inputInfo[0]!!.shape contentEquals inputInfo[1]!!.shape) {
            intArrayOf(it.size)
        } else {
            it.shape
        }
    }

    override fun workGroupSize(inputInfo: List<ArrayInfo?>, outputInfo: List<ArrayInfo?>): IntArray {
        val outputShape = outputShape(inputInfo, outputInfo)
        if (outputShape.size == 1) {
            return intArrayOf(WORK_GROUP_SIZE_1D, 1, 1)
        }
        val reversedShape = outputShape.reversedArray()
        var x = WORK_GROUP_SIZE_1D
        var y = 1
        while (reversedShape[0] * 2 <= x) {
            x /= 2
            y *= 2
        }
        if (outputShape.size == 2) {
            return intArrayOf(x, y, 1)
        }
        var z = 1
        while (reversedShape[1] * 2 <= y) {
            y /= 2
            z *= 2
        }
        return intArrayOf(x, y, z)
    }

    override fun dispatchSize(inputInfo: List<ArrayInfo?>, outputInfo: List<ArrayInfo?>, workGroupSize: IntArray): IntArray =
        (shapeToWorkSize(outputShape(inputInfo, outputInfo)) zip workGroupSize).map { (dim, workSize) -> dim divUp workSize }.toIntArray()

    override fun createShader(inputInfo: List<ArrayInfo?>, outputInfo: List<ArrayInfo?>): String {
        val shapes = arrayListOf(
            inputInfo[0]!!.shape,
            inputInfo[1]!!.shape,
            outputShape(inputInfo, outputInfo)
        )
        if (shapes[0] contentEquals shapes[1]) {
            shapes.fill(intArrayOf(outputInfo[0]!!.size))
        }
        val maxRank = shapes.maxOf { it.size }
        shapes.forEachIndexed { index, shape ->
            shapes[index] = unsqueezeFirst(shape, maxRank)
        }

        val outputShape = shapes.last()
        val outputShapeReversed = outputShape.reversedArray()
        val workGroupSize = workGroupSize(inputInfo, outputInfo)
        val bounds = listOfNotNull(
            outputShapeReversed.elementAtOrNull(0),
            outputShapeReversed.elementAtOrNull(1),
            outputShapeReversed.drop(2).let { if (it.isEmpty()) null else it.fold(1, Int::times) }
        )
        val indices = outputShapeReversed.mapIndexed { index, value ->
            if (index < 2 || outputShapeReversed.size <= 3) {
                "    let i$index = global_id[$index];"
            } else {
                // FIXME
                "    let i$index = (global_id[2] / ${outputShapeReversed.drop(index).fold(1, Int::times)}u) % ${value}u;"
            }
        }.joinToString("\n")

        val dataIndices = shapes.map { shape ->
            Strides(shape).strides.mapIndexed { index, value ->
                if (shape[index] == 1) "0u" else "${value}u * i${maxRank - index - 1}"
            }.joinToString(separator = " + ")
        }
        val (input0, input1, output) = dataIndices.mapIndexed { index, value -> "matrix${index}.data[${value}]" }

        return """
[[block]] struct Matrix {
    data: array<${outputInfo[0]!!.type.wgslType}>;
};

[[group(0), binding(0)]] var<storage, read> matrix0 : Matrix;
[[group(0), binding(1)]] var<storage, read> matrix1 : Matrix;
[[group(0), binding(2)]] var<storage, read_write> matrix2 : Matrix;

[[stage(compute), workgroup_size(${workGroupSize.joinToString()})]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    if (${bounds.withIndex().joinToString(separator = " || ") { (index, value) -> "global_id[$index] >= ${value}u" }}) {
        return;
    }
$indices
    
    ${operation(input0, input1, output)}
}
"""
    }
}
