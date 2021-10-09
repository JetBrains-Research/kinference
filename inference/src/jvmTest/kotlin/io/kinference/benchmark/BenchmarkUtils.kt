package io.kinference.benchmark

import ai.onnxruntime.*
import io.kinference.core.KIEngine
import io.kinference.core.data.tensor.KITensor
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.loadModel
import io.kinference.model.Model
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import java.io.File
import java.nio.*

object BenchmarkUtils {
    private val ortOptions
        get() = OrtSession.SessionOptions().apply {
            this.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            this.setInterOpNumThreads(1)
            this.setIntraOpNumThreads(1)
            this.disableProfiling()
            this.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
        }

    fun KITensor.toOnnxTensor(env: OrtEnvironment) = when (this.data.type) {
        DataType.FLOAT -> OnnxTensor.createTensor(env, FloatBuffer.wrap((data as FloatNDArray).array.toArray()), data.shape.toLongArray())
        DataType.DOUBLE -> OnnxTensor.createTensor(env, DoubleBuffer.wrap((data as DoubleNDArray).array.toArray()), data.shape.toLongArray())
        DataType.INT -> OnnxTensor.createTensor(env, IntBuffer.wrap((data as IntNDArray).array.toArray()), data.shape.toLongArray())
        DataType.LONG -> OnnxTensor.createTensor(env, LongBuffer.wrap((data as LongNDArray).array.toArray()), data.shape.toLongArray())
        else -> throw UnsupportedOperationException()
    }

    private fun IntArray.toLongArray() = LongArray(size) { this[it].toLong() }

    fun modelWithInputs(path: String): Pair<ByteArray, List<KITensor>> {
        val (mainPath, testName, dataSet) = path.split('.')

        val testDir = javaClass.getResource("/$mainPath/test_$testName").path

        val modelBytes = File("$testDir/model.onnx").readBytes()
        val inputFiles = File("$testDir/test_data_set_$dataSet/").listFiles()!!.filter { "input" in it.name }
        val inputs = inputFiles.map { KIEngine.loadData(it.readBytes(), ONNXDataType.ONNX_TENSOR) as KITensor }

        return modelBytes to inputs
    }

    data class OrtState(val session: OrtSession, val inputs: Map<String, OnnxTensor>) {
        companion object {
            fun create(path: String): OrtState {
                val (modelBytes, inputs) = modelWithInputs(path)

                val env = OrtEnvironment.getEnvironment()
                val session = env.createSession(modelBytes, ortOptions)
                val ortInputs = inputs.associate { it.name!! to it.toOnnxTensor(env) }

                return OrtState(session, ortInputs)
            }
        }
    }

    data class KIState(val model: Model<ONNXData<*>>, val inputs: List<KITensor>) {
        companion object {
            fun create(path: String): KIState {
                val (modelBytes, inputs) = modelWithInputs(path)
                val model = KIEngine.loadModel(modelBytes)
                return KIState(model, inputs)
            }
        }
    }
}
