package io.kinference.benchmark.models

import io.kinference.benchmark.BenchmarkUtils.KIState
import io.kinference.benchmark.BenchmarkUtils.OrtState
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

@Warmup(iterations = 30)
@Measurement(iterations = 100)
@State(Scope.Benchmark)
open class PosTaggerBenchmarkKI {
    val path = "pos.pos_tagger.0"
    lateinit var state: KIState

    @Setup(Level.Trial)
    fun setup() {
        state = KIState.create(path)
    }

    @Benchmark
    fun benchmark(blackhole: Blackhole) {
        val outputs = state.model.predict(state.inputs)
        blackhole.consume(outputs)
    }
}

@Warmup(iterations = 30)
@Measurement(iterations = 100)
@State(Scope.Benchmark)
open class PosTaggerBenchmarkORT {
    val path = "pos.pos_tagger.0"
    lateinit var state: OrtState

    @Setup(Level.Trial)
    fun setup() {
        state = OrtState.create(path)
    }

    @Benchmark
    fun benchmark(blackhole: Blackhole) {
        val outputs = state.session.run(state.inputs)
        blackhole.consume(outputs)
    }
}
