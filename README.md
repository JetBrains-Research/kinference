<h1> <img align="center" width="64" height="64" src="https://s3-eu-west-1.amazonaws.com/public-resources.ml-labs.aws.intellij.net/static/kinference/icon_256.png" alt="KInference Icon"> KInference </h1>

[![JB Research](https://jb.gg/badges/research-flat-square.svg)](https://research.jetbrains.org/)

KInference is a library that makes possible execution of complex ML models (written via ONNX) in vanilla Kotlin.

[ONNX](https://github.com/onnx/onnx) is a popular ecosystem for building, training, evaluating, and exchanging ML and DL models. It makes the process much
simpler and divides the model into building blocks that can be switched or tuned to one's liking.

However, ONNX carries with itself a lot of dependencies and requirements that complicate its use in some cases. Our library does not require anything but
vanilla Kotlin. KInference is lightweight but fast, and supports numerous ONNX operators. This makes it easier to use and is especially useful for various
applications that require the models to be run on the users' machines.

Right now, KInference is in active development.

## Setup

Latest version of KInference is: *0.1.20*

In `build.gradle.kts` you should add repository via:

```kotlin
repositories {
    maven {
        url = uri("https://packages.jetbrains.team/maven/p/ki/maven")
    }
}
```

After it you can add KInference Runtime as dependency:

```kotlin
dependencies {
    api("io.kinference", "inference-core", "0.1.20")
}
```
