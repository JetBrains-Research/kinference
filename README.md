<h1> <img align="center" width="64" height="64" src="https://s3-eu-west-1.amazonaws.com/public-resources.ml-labs.aws.intellij.net/static/kinference/icon_256.png" alt="KInference Icon"> KInference </h1>

[![JB Research](https://jb.gg/badges/research-flat-square.svg)](https://research.jetbrains.org/)

KInference is a library that makes possible execution of complex ML models (written via ONNX) in Kotlin.

[ONNX](https://github.com/onnx/onnx) is a popular ecosystem for building, training, evaluating, and exchanging ML and DL models. It makes the process much
simpler and divides the model into building blocks that can be switched or tuned to one's liking.

However, ONNX carries with itself a lot of dependencies and requirements that complicate its use in some cases. 
KInference is designed to facilitate ONNX models inference on a variety of platforms via configurable backends.
Our library addresses the problem of local inference as well and provides several solutions that are suitable for running on users' machines. 

Right now, KInference is in active development.

## KInference backends
### KInference Core
Pure Kotlin implementation that requires anything but vanilla Kotlin. KInference Core is lightweight but fast, and supports numerous ONNX operators.
It makes the library easy to use and especially convenient for various applications that require the models to run locally on users' machines.

### TensorFlow.js
High-performance JavaScript backend that relies on [Tensorflow.js](https://www.tensorflow.org/js/) library. 
Essentially, it employs GPU operations provided by TensorFlow.js to boost the computations. 
In addition, this implementation enables model execution directly in the user's browser.

### ONNXRuntime and ONNXRuntime GPU
Java backends that use [ONNXRuntime](https://github.com/microsoft/onnxruntime) as an inference engine.

## Setup
Latest version of KInference is: *0.2.0*

First, in `build.gradle.kts` you should add KInference repository via:

```kotlin
repositories {
    maven {
        url = uri("https://packages.jetbrains.team/maven/p/ki/maven")
    }
}
```

### KInference Core
To enable the backend, you can add KInference Runtime as dependency:

```kotlin
dependencies {
    api("io.kinference", "inference-core", "0.2.0")
}
```

### Multi-backend project setup
To configure individual KInference backend for each target, you should just add corresponding backends to the dependencies.

```kotlin
kotlin {
    jvm {}

    js(IR) {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("io.kinference:inference-api:0.2.0")
                api("io.kinference:ndarray-api:0.2.0")
            }
        }

        val jvmMain by getting {
            dependencies {
                api("io.kinference:inference-core:0.2.0")
            }
        }

        val jsMain by getting {
            dependencies {
                api("io.kinference:inference-tfjs:0.2.0")
            }
        }
    }
}
```
