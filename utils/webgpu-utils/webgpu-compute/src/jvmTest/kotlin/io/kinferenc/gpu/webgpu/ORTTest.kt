package io.kinferenc.gpu.webgpu

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession.SessionOptions
import org.junit.jupiter.api.Test
import java.nio.FloatBuffer
import kotlin.random.Random
import kotlin.system.measureTimeMillis


class ORTTest {
    @Test
    fun benchmark() {
        val matrixDimension = 1000

        val random = Random(1312)

        val count = 100

        val modelPath: String = this::class.java.getResource("/matmul_test/model.onnx")!!.path
        OrtEnvironment.getEnvironment("benchmarkORT").use { env ->
            SessionOptions().use { options ->
                env.createSession(modelPath, options).use { session ->
                    val times = ArrayList<Long>()
                    repeat(count) {
                        val matrixA = FloatArray(matrixDimension * matrixDimension) { random.nextFloat() * 10 }
                        val matrixB = FloatArray(matrixDimension * matrixDimension) { random.nextFloat() * 10 }
                        val input = mapOf(
                            "x" to OnnxTensor.createTensor(env, FloatBuffer.wrap(matrixA), longArrayOf(matrixDimension.toLong(), matrixDimension.toLong())),
                            "y" to OnnxTensor.createTensor(env, FloatBuffer.wrap(matrixB), longArrayOf(matrixDimension.toLong(), matrixDimension.toLong()))
                        )

                        times += measureTimeMillis {
                            session.run(input)
                        }
                    }
                    //val result = session.run(input)
                    //println((result.get("output").get() as OnnxTensor).floatBuffer.array().take(10))

                    println("Avg time millis: ${times.average()}")
                    println("Times: $times")
                }
            }
        }
    }
}