//@file:OptIn(KotlinxBenchmarkPluginInternalApi::class)

//import org.gradle.kotlin.dsl.support.serviceOf
import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.configureTests
import kotlinx.benchmark.gradle.JmhBytecodeGeneratorTask
import kotlinx.benchmark.gradle.JmhBytecodeGeneratorWorker
import kotlinx.benchmark.gradle.internal.KotlinxBenchmarkPluginInternalApi
import org.gradle.kotlin.dsl.support.serviceOf
import org.gradle.kotlin.dsl.withType
import kotlin.jvm.java

group = rootProject.group
version = rootProject.version

plugins {
    id("org.jetbrains.kotlin.multiplatform")
    alias(libs.plugins.kotlin.atomicfu)
    id("org.jetbrains.kotlinx.benchmark") version "0.4.14"
}

benchmark {
    targets {
        register("jvmBenchmark")
    }

    configurations {
        val names = arrayOf("Bert", "Electra")
        val types = arrayOf("Light", "Heavy")

        val warmup = mapOf("Light" to 3, "Heavy" to 15)
        val iters = mapOf("Light" to 6, "Heavy" to 10)
        val time = mapOf("Light" to 12L, "Heavy" to 30L)

        for (type in types) {
            for (name in names) {
                register(name.lowercase() + type) {
                    include(name)
                    warmups = warmup[type] // number of warmup iterations
                    iterations = iters[type] // number of iterations
                    iterationTime = time[type] // time in seconds per iteration
                    iterationTimeUnit = "SECONDS"
                    reportFormat = "text"
                }
            }
        }
    }
}

kotlin {
    jvm {
        configureTests()
        configureHeavyTests()
        configureBenchmarkTests()
    }

    sourceSets {
        jvmMain {
            dependencies {
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.atomicfu)
                implementation(libs.okio)

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))
                api(project(":inference:inference-ir-trees"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api(project(":serialization:serializer-tiled"))
            }
        }

        all {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.14")
                api(project(":utils:utils-testing"))
            }
            compilerOptions {
                freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector")
            }
        }

        jvmTest {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }

        }

    }
    tasks.withType<JavaExec>().configureEach {
        jvmArgs = listOf("--add-modules=jdk.incubator.vector")
    }
    tasks.withType<Test>().configureEach {
        jvmArgs = listOf("--add-modules=jdk.incubator.vector")
    }

    jvm {
        compilations.create("benchmark") {
            associateWith(this@jvm.compilations.getByName("main"))
        }
    }

    afterEvaluate {
        val workerExecutor = serviceOf<WorkerExecutor>()

        @OptIn(KotlinxBenchmarkPluginInternalApi::class) tasks.withType<JmhBytecodeGeneratorTask> {
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
}

