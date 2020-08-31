package io.kinference.benchmark

import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit
import kotlin.random.Random


@State(Scope.Benchmark)
@Fork(value = 1, warmups = 0, jvmArgsAppend = [
    "-XX:CompileThreshold=100",
    "-XX:+UnlockDiagnosticVMOptions"
//    "-XX:CompileCommand=print,\"io.kinference/benchmark/DotBenchmark.baseline\""
])
@Warmup(iterations = 3)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Measurement(iterations = 100)
open class DotBenchmark {
    @Param("10", "100", "1000")
    var n = 0

    @Param("10", "100", "1000")
    var m = 0

    @Param("10", "100", "1000")
    var t = 0

    lateinit var left: FloatArray
    lateinit var right: FloatArray
    lateinit var dest: FloatArray

    lateinit var array: FloatArray

    var leftSize: Int = 0
    var rightSize: Int = 0
    var destSize: Int = 0

    @Setup(Level.Iteration)
    fun setup() {
        leftSize = n * t
        rightSize = t * m
        destSize = n * m

        left = FloatArray(leftSize) { Random.nextFloat() }
        right = FloatArray(rightSize) { Random.nextFloat() }
        dest = FloatArray(destSize)

        array = FloatArray(destSize)
    }

    @TearDown(Level.Iteration)
    fun teardown() {
        for (i in 0 until destSize) {
            dest[i] = 0f
        }
    }

    @Benchmark
    fun baseline(blackhole: Blackhole) {
        for (i in 0 until n) {
            val dInd = i * m
            val lInd = i * t
            for (k in 0 until t) {
                val temp = left[lInd + k]
                val rInd = k * m
                for (j in 0 until m) {
                    dest[dInd + j] += temp * right[rInd + j]
                }
            }
        }

        blackhole.consume(dest)
    }

    @Benchmark
    fun baselineWithBuffer(blackhole: Blackhole) {
        val buffer = FloatBuffer.wrap(array)

        for (i in 0 until n) {
            val dInd = i * m
            val lInd = i * t
            for (k in 0 until t) {
                val temp = left[lInd + k]

                buffer.position(dInd)
                buffer.put(right, k * m, m)

                for (j in dInd until dInd + m) {
                    dest[j] += temp * array[j]
                }
            }
        }

        blackhole.consume(dest)
    }
}

class BenchmarkTest {
    @Test
    @Tag("benchmark")
    fun `test dot performance`() {
        val opts = OptionsBuilder()
            .include("DotBenchmark")
//            .addProfiler("org.openjdk.jmh.profile.WinPerfAsmProfiler")
            .build()
        val results = Runner(opts).run().toTypedArray()

        assert(results[0].primaryResult.getScore() < 500)
        assert(results[1].primaryResult.getScore() < 150)
    }
}
