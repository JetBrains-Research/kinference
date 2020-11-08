package io.kinference.utils

import com.amazonaws.auth.AWSStaticCredentialsProvider
import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.amazonaws.services.s3.model.*
import com.amazonaws.util.Md5Utils
import java.io.*


object S3Client {
    private const val bucket: String = "kinference.grazie.aws.intellij.net"

    private val client by lazy {
        AmazonS3ClientBuilder.standard()
            .withRegion("eu-west-1")
            .withCredentials(AWSStaticCredentialsProvider(BasicAWSCredentials(Config.awsAccessKey, Config.awsSecretKey)))
            .build()
    }

    fun getObjects(prefix: String): List<S3ObjectSummary> {
        val objects = ArrayList<S3ObjectSummary>()
        var isFirst = true
        var objectListing = client.listObjects(bucket)
        while (objectListing.isTruncated || isFirst) {
            if (!isFirst) {
                objectListing = client.listNextBatchOfObjects(objectListing)
            }
            objects.addAll(objectListing.objectSummaries.filter {
                it.key.startsWith(prefix)
            })
            isFirst = false
        }
        return objects
    }

    fun copyObjects(prefix: String, toFolder: File) {
        val objects = getObjects(prefix).filter { !it.key.endsWith("/") }
        val toDownload = objects.filter {
            val toKey = it.key.drop(prefix.length)
            val toFile = File(toFolder, toKey)

            //Have to use variant with size because of multipart etag in s3
            !toFile.exists() || toFile.length() != it.size
        }

        if (toDownload.isEmpty()) return

        println("Downloading missing files from s3://${bucket}/$prefix")
        for (obj in toDownload) {
            val toKey = obj.key.drop(prefix.length)
            val toFile = File(toFolder, toKey)

            if (toFile.exists()) toFile.delete()

            println("Downloading file from key ${obj.key}")

            toFile.parentFile.mkdirs()

            toFile.outputStream().use { output ->
                getObject(obj.key).objectContent.use { input -> input.transferTo(output) }
            }
        }

        println("Finished files downloading")
    }

    fun getObject(key: String): S3Object {
        return client.getObject(bucket, key)
    }

    fun putObject(key: String, content: ByteArray) {
        client.putObject(bucket, key, content.inputStream(), ObjectMetadata().also {
            it.contentLength = content.size.toLong()
        })
    }

    private fun InputStream.transferTo(output: OutputStream) {
        val buffer = ByteArray(1024)
        var len: Int = read(buffer)
        while (len != -1) {
            output.write(buffer, 0, len)
            len = read(buffer)
        }
    }

    private fun File.md5(): String {
        return Md5Utils.computeMD5Hash(this).joinToString(separator = "") { "%02x".format(it) }
    }
}

