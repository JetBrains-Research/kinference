package io.kinference.webgpu.operators.math

import io.kinference.attribute.Attribute
import io.kinference.ndarray.Strides
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.broadcasting.unsqueezeFirst
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.webgpu.*
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.NDArrayInfo
import io.kinference.webgpu.ndarray.WebGPUDataType
import io.kinference.webgpu.operators.common.*
import io.kinference.webgpu.utils.divUp

sealed class MatMul(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : CachingShaderOperator(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulVer1.VERSION.asRange() -> MatMulVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of MatMul operator: $version")
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

    override fun operatorImplementation(inputInfo: List<NDArrayInfo?>, context: WebGPUContext): Operator<WebGPUTensor, WebGPUTensor> =
        when {
            inputInfo[1]!!.shape.size >= 2 && inputInfo[1]!!.shape.takeLast(2).all { it % 4 == 0 } -> MatMulPackedVec4(
                context.gpuState.device, inputInfo[0]!!, inputInfo[1]!!,
                info = info, attributes = attributes, inputs = inputs, outputs = outputs
            )
            else -> MatMulPackedUnaligned(
                context.gpuState.device, inputInfo[0]!!, inputInfo[1]!!,
                info = info, attributes = attributes, inputs = inputs, outputs = outputs
            )
        }
}

abstract class MatMulPacked(
    device: Device,
    private val inputInfo0: NDArrayInfo,
    private val inputInfo1: NDArrayInfo,
    info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : BinaryOperator(device, info, attributes, inputs, outputs) {

    override val outputShape: IntArray = Broadcasting.broadcastShapeForMatmul(inputInfo0.shape, inputInfo1.shape)
    override val outputType: WebGPUDataType = inputInfo0.type

    protected abstract val threadWorkX: Int
    protected abstract val threadWorkY: Int

    protected open val vecSize = 1

    private val workSize: IntArray
        get() = intArrayOf(workGroupSize[0] * threadWorkX, workGroupSize[1] * threadWorkY, workGroupSize[2])

    override val dispatchSize: IntArray
        get() = (shapeToWorkSize(outputInfo.shape) zip workSize).map { (dim, workSize) -> dim divUp workSize }.toIntArray()

    private val shapes: List<IntArray> = kotlin.run {
        val shapes = mutableListOf(inputInfo0.shape, inputInfo1.shape, outputInfo.shape)
        if (shapes[1].size == 1) {
            shapes[1] = shapes[1] + 1
        }
        val maxRank = maxOf(shapes.maxOf { it.size }, 3)
        shapes.forEachIndexed { index, shape ->
            shapes[index] = unsqueezeFirst(shape, maxRank)
        }
        shapes
    }

    protected val m = shapes[0].takeLast(2).first()
    protected val n = shapes[0].last()
    protected val k = shapes[1].last()

    private val outerShapes = shapes.map { it.dropLast(2).toIntArray() }

    protected val indices = kotlin.run {
        val outerShapeReversed = outerShapes.last().reversedArray()
        outerShapeReversed.mapIndexed { index, value ->
            "    let i$index = (i32(global_id[2]) / ${outerShapeReversed.take(index).fold(1, Int::times)}) % ${value};"
        }.joinToString(separator = "\n")
    }

    protected val dataOffsets
        get() = outerShapes.map { shape ->
            Strides(shape).strides.mapIndexed { index, value ->
                if (shape[index] == 1) "0" else "$value * i${shape.size - index - 1}"
            }.joinToString(separator = " + ")
        }.withIndex().joinToString(separator = "\n") { (index, value) ->
            "    let offset$index = ($value) * ${shapes[index].takeLast(2).fold(1, Int::times) / vecSize};"
        }
}

class MatMulPackedVec4(
    device: Device,
    inputInfo0: NDArrayInfo,
    inputInfo1: NDArrayInfo,
    info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : MatMulPacked(device, inputInfo0, inputInfo1, info, attributes, inputs, outputs) {
    companion object {
        const val VEC_SIZE = 4
    }

    override val threadWorkX: Int = VEC_SIZE
    override val threadWorkY: Int = VEC_SIZE

    override val vecSize = VEC_SIZE

    override val shader: String
        get() {
            val wgslType = outputInfo.type.wgslType
            val tileSizeX = workGroupSize[0]
            val tileSizeY = workGroupSize[1] * threadWorkY

            return """
struct Matrix {
    data: array<vec4<$wgslType>>;
};

[[group(0), binding(0)]] var<storage, read> matrixA : Matrix;
[[group(0), binding(1)]] var<storage, read> matrixB : Matrix;
[[group(0), binding(2)]] var<storage, read_write> matrixC : Matrix;

var<workgroup> localA : array<array<vec4<$wgslType>, $tileSizeX>, $tileSizeY>;
var<workgroup> localB : array<array<vec4<$wgslType>, $tileSizeX>, $tileSizeY>;

[[stage(compute), workgroup_size(${workGroupSize.joinToString()})]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>,
        [[builtin(local_invocation_id)]] local_id : vec3<u32>) {
    let global_row = i32(global_id.y) * $threadWorkY;
    let global_col = i32(global_id.x);
    
    let tile_row = i32(local_id.y) * $threadWorkY;
    let tile_col = i32(local_id.x);
    
$indices
$dataOffsets
    
    var cachedA: vec4<$wgslType>;
    var cachedB: array<vec4<$wgslType>, $threadWorkY>;
    var acc: array<vec4<$wgslType>, $threadWorkY>;
    
    for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
        acc[inner_row] = vec4<$wgslType>($wgslType(0));
    }

    for (var start = 0; start < ${n / vecSize}; start = start + $tileSizeX) {
        for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
            if (global_row + inner_row < $m && start + tile_col < ${n / vecSize}) {
                localA[tile_row + inner_row][tile_col] = matrixA.data[offset0 + (global_row + inner_row) * ${n / vecSize} + (start + tile_col)];
            } else {
                localA[tile_row + inner_row][tile_col] = vec4<$wgslType>($wgslType(0));
            }
        }
        for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
            if (start * $threadWorkY + tile_row + inner_row < $n && global_col < ${k / vecSize}) {
                localB[tile_row + inner_row][tile_col] = matrixB.data[offset1 + (start * $vecSize + tile_row + inner_row) * ${k / vecSize} + global_col];
            } else {
                localB[tile_row + inner_row][tile_col] = vec4<$wgslType>($wgslType(0));
            }
        }
        workgroupBarrier();

        for (var k = 0; k < ${workGroupSize[0]}; k = k + 1) {
            cachedB[0] = localB[k * $threadWorkY][tile_col];
            cachedB[1] = localB[k * $threadWorkY + 1][tile_col];
            cachedB[2] = localB[k * $threadWorkY + 2][tile_col];
            cachedB[3] = localB[k * $threadWorkY + 3][tile_col];
            for (var i = 0; i < $threadWorkY; i = i + 1) {
                cachedA = localA[tile_row + i][k];
                acc[i] = cachedB[0] * cachedA.x + acc[i];
                acc[i] = cachedB[1] * cachedA.y + acc[i];
                acc[i] = cachedB[2] * cachedA.z + acc[i];
                acc[i] = cachedB[3] * cachedA.w + acc[i];
            }
        }
        workgroupBarrier();
    }
    
    for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
        if (global_row + inner_row < $m && global_col < ${k / vecSize}) {
            matrixC.data[offset2 + (global_row + inner_row) * ${k / vecSize} + global_col] = acc[inner_row];
        }
    }
}
"""
        }

    override val workGroupSize: IntArray = intArrayOf(8, 8, 1)
}

class MatMulPackedUnaligned(
    device: Device,
    inputInfo0: NDArrayInfo,
    inputInfo1: NDArrayInfo,
    info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : MatMulPacked(device, inputInfo0, inputInfo1, info, attributes, inputs, outputs) {
    override val threadWorkX: Int = 4
    override val threadWorkY: Int = 4

    override val shader: String
        get() {
            val wgslType = outputInfo.type.wgslType
            val tileSizeX = workGroupSize[0] * threadWorkX
            val tileSizeY = workGroupSize[1] * threadWorkY

            return """
struct Matrix {
    data: array<$wgslType>;
};

[[group(0), binding(0)]] var<storage, read> matrixA : Matrix;
[[group(0), binding(1)]] var<storage, read> matrixB : Matrix;
[[group(0), binding(2)]] var<storage, read_write> matrixC : Matrix;

var<workgroup> localA : array<array<$wgslType, $tileSizeX>, $tileSizeY>;
var<workgroup> localB : array<array<$wgslType, $tileSizeX>, $tileSizeY>;

[[stage(compute), workgroup_size(${workGroupSize.joinToString()})]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>,
        [[builtin(local_invocation_id)]] local_id : vec3<u32>) {
    let global_row = i32(global_id.y) * $threadWorkY;
    let global_col = i32(global_id.x) * $threadWorkX;
    
    let tile_row = i32(local_id.y) * $threadWorkY;
    let tile_col = i32(local_id.x) * $threadWorkX;
    
$indices
$dataOffsets
    
    var cachedA: $wgslType;
    var cachedB: array<$wgslType, $threadWorkY>;
    var acc: array<array<$wgslType, $threadWorkX>, $threadWorkY>;
    
    for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
        for (var inner_col = 0; inner_col < $threadWorkX; inner_col = inner_col + 1) {
            acc[inner_row][inner_col] = $wgslType(0);
        }
    }

    for (var start = 0; start < $n; start = start + $tileSizeX) {
        for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
            let local_row = tile_row + inner_row;
            for (var inner_col = 0; inner_col < $threadWorkX; inner_col = inner_col + 1) {
                let local_col = tile_col + inner_col;
                if (global_row + inner_row < $m && start + local_col < $n) {
                    localA[local_row][local_col] = matrixA.data[offset0 + (global_row + inner_row) * $n + (start + local_col)];
                } else {
                    localA[local_row][local_col] = $wgslType(0);
                }
            }
        }
        for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
            let local_row = tile_row + inner_row;
            for (var inner_col = 0; inner_col < $threadWorkX; inner_col = inner_col + 1) {
                let local_col = tile_col + inner_col;
                if (start + local_row < $n && global_col + inner_col < $k) {
                    localB[local_row][local_col] = matrixB.data[offset1 + (start + local_row) * $k + (global_col + inner_col)];
                } else {
                    localB[local_row][local_col] = $wgslType(0);
                }
            }
        }
        workgroupBarrier();

        for (var k = 0; k < $tileSizeX; k = k + 1) {
            for (var inner = 0; inner < $threadWorkX; inner = inner + 1) {
                cachedB[inner] = localB[k][tile_col + inner];
            }
            for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
                cachedA = localA[tile_row + inner_row][k];
                for (var inner_col = 0; inner_col < $threadWorkX; inner_col = inner_col + 1) {
                    acc[inner_row][inner_col] = acc[inner_row][inner_col] + cachedA * cachedB[inner_col];
                }
            }
        }
        workgroupBarrier();
    }
    
    for (var inner_row = 0; inner_row < $threadWorkY; inner_row = inner_row + 1) {
        for (var inner_col = 0; inner_col < $threadWorkX; inner_col = inner_col + 1) {
            if (global_row + inner_row < $m && global_col + inner_col < $k) {
                matrixC.data[offset2 + (global_row + inner_row) * $k + (global_col + inner_col)] = acc[inner_row][inner_col];
            }
        }
    }
}
"""
        }

    override val workGroupSize: IntArray = intArrayOf(8, 8, 1)
}
