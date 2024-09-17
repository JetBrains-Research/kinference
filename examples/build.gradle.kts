group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":inference:inference-core"))
                api(project(":serialization:serializer-protobuf"))
                api(project(":utils:utils-common"))

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(libs.wire.runtime)
                implementation("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.5.2")
                implementation("org.jetbrains.kotlinx:kotlin-deeplearning-dataset:0.5.2")  // Dataset support

                implementation("io.ktor:ktor-client-core:2.3.12")
                implementation("io.ktor:ktor-client-cio:2.3.12") // JVM Engine

                api("org.slf4j:slf4j-api:2.0.9")
                api("org.slf4j:slf4j-simple:2.0.9")

                implementation("com.knuddels:jtokkit:1.1.0")
            }
        }
    }
}
