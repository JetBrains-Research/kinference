package org.jetbrains.research.kotlin.inference.benchmark.models

import org.jetbrains.research.kotlin.inference.BenchmarkUtils
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

@Warmup(iterations = 10)
@Measurement(iterations = 50)
@State(Scope.Benchmark)
open class GPTBenchmarkKI {
    val path = "gpt.dummy_input.batch_size_8"
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

@Warmup(iterations = 10)
@Measurement(iterations = 50)
@State(Scope.Benchmark)
open class GPTBenchmarkORT {
    val path = "gpt.dummy_input.batch_size_8"
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
