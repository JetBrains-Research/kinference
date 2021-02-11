package io.kinference.benchmark.models

import io.kinference.benchmark.BenchmarkUtils
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import java.util.concurrent.TimeUnit

@Fork(value = 1, warmups = 0)
@BenchmarkMode(Mode.SingleShotTime)
@Warmup(iterations = 10)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Measurement(iterations = 50)
@State(Scope.Benchmark)
open class GPTBenchmarkKI {
    val path = "gpt.dummy_input.1"
    lateinit var state: BenchmarkUtils.KIState

    @Setup(Level.Trial)
    fun setup() {
        state = BenchmarkUtils.KIState.create(path)
    }

    @Benchmark
    fun benchmark(blackhole: Blackhole) {
        val outputs = state.model.predict(state.inputs)
        blackhole.consume(outputs)
    }
}

@Fork(value = 1, warmups = 0)
@BenchmarkMode(Mode.SingleShotTime)
@Warmup(iterations = 10)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Measurement(iterations = 50)
@State(Scope.Benchmark)
open class GPTBenchmarkORT {
    val path = "gpt.dummy_input.1"
    lateinit var state: BenchmarkUtils.OrtState

    @Setup(Level.Trial)
    fun setup() {
        state = BenchmarkUtils.OrtState.create(path)
    }

    @Benchmark
    fun benchmark(blackhole: Blackhole) {
        val outputs = state.session.run(state.inputs)
        blackhole.consume(outputs)
    }
}

class GPTBenchmark {
    @Test
    fun benchmark_gpt_performance() {
        val opts = OptionsBuilder()
            .include("GPTBenchmark*")
            .jvmArgsAppend("-Xmx2000m")
            .build()

        Runner(opts).run()
    }
}
