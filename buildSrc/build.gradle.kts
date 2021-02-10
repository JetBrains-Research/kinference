repositories {
    mavenCentral()
}

plugins {
    `kotlin-dsl`
}

dependencies {
    implementation(gradleApi())
    api("com.amazonaws:aws-java-sdk-s3:1.11.896")
}
