@file:OptIn(KotlinxBenchmarkPluginInternalApi::class)

import kotlinx.benchmark.gradle.JmhBytecodeGeneratorTask
import kotlinx.benchmark.gradle.JmhBytecodeGeneratorWorker
import kotlinx.benchmark.gradle.internal.KotlinxBenchmarkPluginInternalApi
import org.gradle.kotlin.dsl.support.serviceOf

group = rootProject.group
version = rootProject.version

val enableVectorization = project.findProperty("enableVectorization")?.toString()?.toBoolean() ?: false

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
            warmups = 5 // number of warmup iterations
            iterations = 15 // number of iterations
            iterationTime = 10 // time in seconds per iteration
            iterationTimeUnit = "SECONDS"
            reportFormat = "text"
        }

        var types = arrayOf("Double", "Float")
        val benchmarks = arrayOf("Dot", "Elu", "Exp", "Gelu", "Logistic", "Neg", "Probit", "Softmax")
        for (type in types)
            register(type.lowercase()) {
                include(".*$type.*")
                warmups = 3 // number of warmup iterations
                iterations = 5 // number of iterations
                iterationTime = 10 // time in seconds per iteration
                iterationTimeUnit = "SECONDS"
                reportFormat = "text"
            }

        types += ""
        for (type in types)
            for (name in benchmarks)
                register(type.lowercase() + name) {
                    include(".*$type$name.*")
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

if (enableVectorization) {
    primitives {
        vectorize = true
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
}
