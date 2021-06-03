package io.kinference.benchmark.models

import io.kinference.benchmark.BenchmarkUtils.KIState
import io.kinference.benchmark.BenchmarkUtils.OrtState
import org.junit.jupiter.api.*
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import java.util.concurrent.TimeUnit

@Fork(value = 1, warmups = 0, jvmArgsAppend = [
    "-XX:CompileThreshold=100",
    "-XX:+UnlockDiagnosticVMOptions"
//    "-XX:CompileCommand=print,\"io.kinference/benchmark/DotBenchmark.baseline\""
])
@BenchmarkMode(Mode.SingleShotTime)
@Warmup(iterations = 3)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
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

@Fork(value = 1, warmups = 0)
@BenchmarkMode(Mode.SingleShotTime)
@Warmup(iterations = 3)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
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

@Disabled
class PosTaggerBenchmark {
    @Test
    fun benchmark_pos_tagger_performance() {
        val opts = OptionsBuilder()
            .include("PosTaggerBenchmark*")
            .build()

        Runner(opts).run()
    }
}
