package io.kinference.webgpu.operators.math

import io.kinference.attribute.Attribute
import io.kinference.ndarray.Strides
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.broadcasting.unsqueezeFirst
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.ndarray.ArrayInfo
import io.kinference.webgpu.operators.common.*
import io.kinference.webgpu.utils.WORK_GROUP_SIZE_2D
import io.kinference.webgpu.utils.divUp

sealed class MatMul(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : BinaryOperator(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulVer1.VERSION.asRange() -> MatMulVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Add operator: $version")
        }
    }
}

class MatMulVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMul(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("MatMul", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun outputInfo(inputInfo: List<ArrayInfo?>): List<ArrayInfo?> =
        listOf(ArrayInfo(Broadcasting.broadcastShapeForMatmul(inputInfo[0]!!.shape, inputInfo[1]!!.shape), inputInfo[0]!!.type))

    override fun workGroupSize(inputInfo: List<ArrayInfo?>, outputInfo: List<ArrayInfo?>): IntArray =
        intArrayOf(WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D, 1)

    override fun dispatchSize(inputInfo: List<ArrayInfo?>, outputInfo: List<ArrayInfo?>, workGroupSize: IntArray): IntArray =
        (shapeToWorkSize(outputInfo.first()!!.shape) zip workGroupSize).map { (dim, workSize) -> dim divUp workSize }.toIntArray()

    override fun createShader(inputInfo: List<ArrayInfo?>, outputInfo: List<ArrayInfo?>): String {
        val shapes = mutableListOf(inputInfo[0]!!.shape, inputInfo[1]!!.shape, outputInfo[0]!!.shape)
        if (shapes[1].size == 1) {
            shapes[1] = shapes[1] + 1
        }
        val maxRank = maxOf(shapes.maxOf { it.size }, 3)
        shapes.forEachIndexed { index, shape ->
            shapes[index] = unsqueezeFirst(shape, maxRank)
        }
        val outerShapes = shapes.map { it.dropLast(2).toIntArray() }.toMutableList()

        val outerShapeReversed = outerShapes.last().reversedArray()
        val indices = outerShapeReversed.mapIndexed { index, value ->
            "    let i$index = (global_id[2] / ${outerShapeReversed.drop(index + 1).fold(1, Int::times)}u) % ${value}u;"
        }.joinToString(separator = "\n")

        val dataOffsets = outerShapes.map { shape ->
            Strides(shape).strides.mapIndexed { index, value ->
                if (shape[index] == 1) "0u" else "${value}u * i${(maxRank - 2) - index - 1}"
            }.joinToString(separator = " + ")
        }.withIndex().joinToString(separator = "\n") { (index, value) ->
            "    let offset$index = ($value) * ${shapes[index].takeLast(2).fold(1, Int::times)}u;"
        }

        val leftShape = shapes[0].takeLast(2)
        val rightShape = shapes[1].takeLast(2)
        val m = leftShape.first()
        val n = leftShape.last().apply { require(this == rightShape.first()) }
        val k = rightShape.last()

        val wgslType = outputInfo[0]!!.type.wgslType

        println(
            """
[[block]] struct Matrix {
    data: array<$wgslType>;
};

[[group(0), binding(0)]] var<storage, read> matrix0 : Matrix;
[[group(0), binding(1)]] var<storage, read> matrix1 : Matrix;
[[group(0), binding(2)]] var<storage, read_write> matrix2 : Matrix;

var<workgroup> localMatrix0 : array<array<f32, $WORK_GROUP_SIZE_2D>, $WORK_GROUP_SIZE_2D>;
var<workgroup> localMatrix1 : array<array<f32, $WORK_GROUP_SIZE_2D>, $WORK_GROUP_SIZE_2D>;

[[stage(compute), workgroup_size($WORK_GROUP_SIZE_2D, $WORK_GROUP_SIZE_2D)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>,
        [[builtin(local_invocation_id)]] local_id : vec3<u32>) {
    var result: $wgslType = $wgslType(0);
    
$indices
$dataOffsets
    
    for (var start = 0u; start < ${n}u; start = start + ${WORK_GROUP_SIZE_2D}u) {
        if (global_id.y < ${m}u && (start + local_id.x) < ${n}u) {
            localMatrix0[local_id.y][local_id.x] = matrix0.data[offset0 + global_id.y * ${n}u + (start + local_id.x)];
        } else {
            localMatrix0[local_id.y][local_id.x] = $wgslType(0);
        }
        if ((start + local_id.y) < ${n}u && global_id.x < ${k}u) {
            localMatrix1[local_id.y][local_id.x] = matrix1.data[offset1 + (start + local_id.y) * ${k}u + global_id.x];
        } else {
            localMatrix1[local_id.y][local_id.x] = $wgslType(0);
        }
        workgroupBarrier();
        
        for (var index = 0u; index < ${WORK_GROUP_SIZE_2D}u; index = index + 1u) {
            result = result + localMatrix0[local_id.y][index] * localMatrix1[index][local_id.x];
        }
        workgroupBarrier();
    }
    
    if (global_id.y < ${m}u && global_id.x < ${k}u) {
        matrix2.data[offset2 + global_id.y * ${k}u + global_id.x] = result;
    }
}
"""
        )

        return """
[[block]] struct Matrix {
    data: array<$wgslType>;
};

[[group(0), binding(0)]] var<storage, read> matrix0 : Matrix;
[[group(0), binding(1)]] var<storage, read> matrix1 : Matrix;
[[group(0), binding(2)]] var<storage, read_write> matrix2 : Matrix;

var<workgroup> localMatrix0 : array<array<f32, $WORK_GROUP_SIZE_2D>, $WORK_GROUP_SIZE_2D>;
var<workgroup> localMatrix1 : array<array<f32, $WORK_GROUP_SIZE_2D>, $WORK_GROUP_SIZE_2D>;

[[stage(compute), workgroup_size($WORK_GROUP_SIZE_2D, $WORK_GROUP_SIZE_2D)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>,
        [[builtin(local_invocation_id)]] local_id : vec3<u32>) {
    var result: $wgslType = $wgslType(0);
    
$indices
$dataOffsets
    
    for (var start = 0u; start < ${n}u; start = start + ${WORK_GROUP_SIZE_2D}u) {
        if (global_id.y < ${m}u && (start + local_id.x) < ${n}u) {
            localMatrix0[local_id.y][local_id.x] = matrix0.data[offset0 + global_id.y * ${n}u + (start + local_id.x)];
        } else {
            localMatrix0[local_id.y][local_id.x] = $wgslType(0);
        }
        if ((start + local_id.y) < ${n}u && global_id.x < ${k}u) {
            localMatrix1[local_id.y][local_id.x] = matrix1.data[offset1 + (start + local_id.y) * ${k}u + global_id.x];
        } else {
            localMatrix1[local_id.y][local_id.x] = $wgslType(0);
        }
        workgroupBarrier();
        
        for (var index = 0u; index < ${WORK_GROUP_SIZE_2D}u; index = index + 1u) {
            result = result + localMatrix0[local_id.y][index] * localMatrix1[index][local_id.x];
        }
        workgroupBarrier();
    }
    
    if (global_id.y < ${m}u && global_id.x < ${k}u) {
        matrix2.data[offset2 + global_id.y * ${k}u + global_id.x] = result;
    }
}
"""
    }
}
