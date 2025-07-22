@file:OptIn(KotlinxBenchmarkPluginInternalApi::class)

import kotlinx.benchmark.gradle.JmhBytecodeGeneratorTask
import kotlinx.benchmark.gradle.JmhBytecodeGeneratorWorker
import kotlinx.benchmark.gradle.internal.KotlinxBenchmarkPluginInternalApi
import org.gradle.kotlin.dsl.support.serviceOf

group = rootProject.group
version = rootProject.version

plugins {
    alias(libs.plugins.kinference.primitives) apply true
    alias(libs.plugins.kotlin.atomicfu)
    id("org.jetbrains.kotlinx.benchmark") version "0.4.14"
}

benchmark {
    targets {
        register("jvmBenchmark")
    }
    configurations {
        register("all") {
            include(".*")
            warmups = 3 // number of warmup iterations
            iterations = 5 // number of iterations
            iterationTime = 10 // time in seconds per iteration
            iterationTimeUnit = "SECONDS"
            reportFormat = "text"
        }
        register("dot") {
            include(".*Dot.*")
            warmups = 3 // number of warmup iterations
            iterations = 5 // number of iterations
            iterationTime = 10 // time in seconds per iteration
            iterationTimeUnit = "SECONDS"
            reportFormat = "text"
        }
        register("softmax") {
            include(".*Softmax.*")
            warmups = 3 // number of warmup iterations
            iterations = 5 // number of iterations
            iterationTime = 10 // time in seconds per iteration
            iterationTimeUnit = "SECONDS"
            reportFormat = "text"
        }
        register("double") {
            include("Double.*")
            warmups = 3 // number of warmup iterations
            iterations = 5 // number of iterations
            iterationTime = 10 // time in seconds per iteration
            iterationTimeUnit = "SECONDS"
            reportFormat = "text"
        }
        register("float") {
            include("Float.*")
            warmups = 3 // number of warmup iterations
            iterations = 5 // number of iterations
            iterationTime = 10 // time in seconds per iteration
            iterationTimeUnit = "SECONDS"
            reportFormat = "text"
        }
    }

}

kotlin {
    jvm()

    sourceSets {
        all {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api(libs.kinference.primitives.annotations)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.atomicfu)
                api(libs.apache.commons.math4.core)
                api(libs.fastutil.core)
                implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.14")
            }
        }
    }
}

kotlin {
    jvm {
        compilations.create("benchmark") {
            associateWith(this@jvm.compilations.getByName("main"))
        }
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    compilerOptions {
        freeCompilerArgs.addAll(
            listOf(
                "-Xadd-modules=jdk.incubator.vector"
            )
        )
    }
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs = listOf("--add-modules=jdk.incubator.vector")
}

afterEvaluate {
    val workerExecutor = serviceOf<WorkerExecutor>()

    @OptIn(KotlinxBenchmarkPluginInternalApi::class)
    tasks.withType<JmhBytecodeGeneratorTask> {
        // Remove the existing @TaskAction
        actions.clear()

        // Custom task action, with vector module
        doLast {
            val workQueue = workerExecutor.processIsolation {
                classpath.setFrom(runtimeClasspath.files)
                if (executableProvider.isPresent) {
                    forkOptions.executable = executableProvider.get()
                }
                // Add required argument:
                forkOptions.jvmArgs = listOf("--add-modules", "jdk.incubator.vector")
            }

            workQueue.submit(JmhBytecodeGeneratorWorker::class.java) {
                inputClasses.setFrom(inputClassesDirs.files)
                inputClasspath.setFrom(inputCompileClasspath.files)
                outputSourceDirectory.set(outputSourcesDir)
                outputResourceDirectory.set(outputResourcesDir)
            }

            workQueue.await()
        }
    }
}

/*
Benchmark                            (rank)   Mode  Cnt      Score      Error  Units
        DoubleDotBenchmark.linearNDArrayDot     100  thrpt    5   4293.235 ±  192.413  ops/s
        DoubleDotBenchmark.linearNDArrayDot     400  thrpt    5    175.286 ±    6.367  ops/s
        DoubleDotBenchmark.linearNDArrayDot    1000  thrpt    5     10.312 ±    0.837  ops/s
        DoubleDotBenchmark.parallelLVDot        100  thrpt    5   4490.225 ±  279.266  ops/s
        DoubleDotBenchmark.parallelLVDot        400  thrpt    5    430.896 ±   63.671  ops/s
        DoubleDotBenchmark.parallelLVDot       1000  thrpt    5     24.269 ±    0.145  ops/s
        DoubleDotBenchmark.standardDot          100  thrpt    5  17323.375 ±  727.655  ops/s
        DoubleDotBenchmark.standardDot          400  thrpt    5    696.071 ±   41.550  ops/s
        DoubleDotBenchmark.standardDot         1000  thrpt    5     41.240 ±    6.014  ops/s
        DoubleSoftmaxBenchmark.blkVecSM         100  thrpt    5    421.410 ±    9.971  ops/s
        DoubleSoftmaxBenchmark.blkVecSM         200  thrpt    5     79.556 ±    0.614  ops/s
        DoubleSoftmaxBenchmark.blkVecSM         400  thrpt    5     11.867 ±    0.154  ops/s
        DoubleSoftmaxBenchmark.linVecSM         100  thrpt    5   1338.402 ±   24.103  ops/s
        DoubleSoftmaxBenchmark.linVecSM         200  thrpt    5    128.941 ±    1.366  ops/s
        DoubleSoftmaxBenchmark.linVecSM         400  thrpt    5      6.918 ±    0.044  ops/s
        DoubleSoftmaxBenchmark.standardSM       100  thrpt    5    347.783 ±    4.216  ops/s
        DoubleSoftmaxBenchmark.standardSM       200  thrpt    5     51.856 ±    0.134  ops/s
        DoubleSoftmaxBenchmark.standardSM       400  thrpt    5      7.023 ±    0.063  ops/s
        FloatDotBenchmark.linearNDArrayDot      100  thrpt    5   4815.298 ±  156.322  ops/s
        FloatDotBenchmark.linearNDArrayDot      400  thrpt    5    177.058 ±    0.739  ops/s
        FloatDotBenchmark.linearNDArrayDot     1000  thrpt    5     10.513 ±    0.047  ops/s
        FloatDotBenchmark.parallelLVDot         100  thrpt    5   6500.916 ±  147.121  ops/s
        FloatDotBenchmark.parallelLVDot         400  thrpt    5    849.203 ±   60.019  ops/s
        FloatDotBenchmark.parallelLVDot        1000  thrpt    5     42.178 ±    0.529  ops/s
        FloatDotBenchmark.standardDot           100  thrpt    5  17300.875 ± 1710.093  ops/s
        FloatDotBenchmark.standardDot           400  thrpt    5    849.870 ±   31.236  ops/s
        FloatDotBenchmark.standardDot          1000  thrpt    5     86.744 ±    1.311  ops/s
        FloatSoftmaxBenchmark.blkVecSM          100  thrpt    5    594.085 ±    1.938  ops/s
        FloatSoftmaxBenchmark.blkVecSM          200  thrpt    5    106.335 ±    0.811  ops/s
        FloatSoftmaxBenchmark.blkVecSM          400  thrpt    5     14.395 ±    0.159  ops/s
        FloatSoftmaxBenchmark.linVecSM          100  thrpt    5   3194.941 ±   34.020  ops/s
        FloatSoftmaxBenchmark.linVecSM          200  thrpt    5    379.616 ±    1.096  ops/s
        FloatSoftmaxBenchmark.linVecSM          400  thrpt    5     31.007 ±    0.374  ops/s
        FloatSoftmaxBenchmark.standardSM        100  thrpt    5    362.751 ±    4.657  ops/s
        FloatSoftmaxBenchmark.standardSM        200  thrpt    5     55.521 ±    0.495  ops/s
        FloatSoftmaxBenchmark.standardSM        400  thrpt    5      7.273 ±    0.176  ops/s
*/

/*
Capped parallelization
DoubleSoftmaxBenchmark.blkVecSM       100  thrpt    3   500.802 ±  25.775  ops/s
DoubleSoftmaxBenchmark.blkVecSM       200  thrpt    3    80.628 ±  34.706  ops/s
DoubleSoftmaxBenchmark.blkVecSM       400  thrpt    3    11.941 ±   0.435  ops/s
DoubleSoftmaxBenchmark.linVecSM       100  thrpt    3   574.314 ±  47.663  ops/s
DoubleSoftmaxBenchmark.linVecSM       200  thrpt    3   110.125 ±  41.282  ops/s
DoubleSoftmaxBenchmark.linVecSM       400  thrpt    3    14.118 ±   7.315  ops/s
DoubleSoftmaxBenchmark.standardSM     100  thrpt    3   397.228 ±  70.742  ops/s
DoubleSoftmaxBenchmark.standardSM     200  thrpt    3    67.053 ±  27.368  ops/s
DoubleSoftmaxBenchmark.standardSM     400  thrpt    3     9.227 ±   0.311  ops/s
FloatSoftmaxBenchmark.blkVecSM        100  thrpt    3   897.749 ±  63.442  ops/s
FloatSoftmaxBenchmark.blkVecSM        200  thrpt    3   161.200 ±  16.981  ops/s
FloatSoftmaxBenchmark.blkVecSM        400  thrpt    3    21.965 ±   1.383  ops/s
FloatSoftmaxBenchmark.linVecSM        100  thrpt    3  3353.511 ± 319.814  ops/s
FloatSoftmaxBenchmark.linVecSM        200  thrpt    3   355.219 ±  36.140  ops/s
FloatSoftmaxBenchmark.linVecSM        400  thrpt    3    31.427 ±   1.477  ops/s
FloatSoftmaxBenchmark.standardSM      100  thrpt    3   573.845 ±  44.445  ops/s
FloatSoftmaxBenchmark.standardSM      200  thrpt    3    93.920 ±   4.832  ops/s
FloatSoftmaxBenchmark.standardSM      400  thrpt    3    11.466 ±   2.524  ops/s
 */
