package com.example.aldifruitcounter

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.Manifest
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var captureButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: Interpreter
    private val cameraRequestCode = 100
    private val storageRequestCode = 200

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        captureButton = findViewById(R.id.captureButton)
        resultTextView = findViewById(R.id.resultTextView)

        // Load TensorFlow Lite model
        tflite = Interpreter(loadModelFile())

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), storageRequestCode)
        }

        // Initialize Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), cameraRequestCode)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = androidx.camera.core.Preview.Builder().build()
                .also { it.setSurfaceProvider(viewFinder.surfaceProvider) }

            imageCapture = ImageCapture.Builder().build()

            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
        }, ContextCompat.getMainExecutor(this))

        captureButton.setOnClickListener { captureAndAnalyzeImage() }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("new_set.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    private fun captureAndAnalyzeImage() {
        val outputFileOptions = ImageCapture.OutputFileOptions.Builder(createTempFile()).build()
        imageCapture.takePicture(outputFileOptions, ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val bitmap = outputFileResults.savedUri?.let { loadBitmapFromFile(it) } // Convert to Bitmap
                    if (bitmap != null) {
                        analyzeImage(bitmap)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("FruitCounter", "Image capture failed: ${exception.message}", exception)
                }
            })
    }

    private fun loadBitmapFromFile(uri: Uri): Bitmap {
        val source = ImageDecoder.createSource(contentResolver, uri)
        return ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            // Force the bitmap to be decoded with software rendering and ARGB_8888 format
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE)
        }
    }



    private fun analyzeImage(bitmap: Bitmap) {
        // Resize bitmap to match model input size (640x640)
        val inputImage = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        // Convert to ByteBuffer
        val inputBuffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4) // 640x640x3, 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder())  // Set correct byte order
        inputBuffer.rewind()

        for (y in 0 until 640) {
            for (x in 0 until 640) {
                val pixel = inputImage.getPixel(x, y)
                inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f) // Red
                inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255.0f)  // Green
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)       // Blue
            }
        }

        // Run inference
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 67, 8400), DataType.FLOAT32) // Output shape [1, 67, 8400]
        tflite.run(inputBuffer, outputBuffer.buffer.rewind())  // Running inference

        // Get the result from the output buffer
        val result = outputBuffer.floatArray

        // Generate a string representation of the result matrix
        val resultString = StringBuilder()

        // Loop through the 67 classes and 8400 values in each class
        for (i in 0 until 67) {
            resultString.append("Class $i:\n")
            for (j in 0 until 8400) {
                val value = result[i * 8400 + j]  // Get the value at position (i, j)
                resultString.append(String.format("%.2f ", value))  // Format value as a string
            }
            resultString.append("\n\n")  // Separate each class with a newline
        }

        // Show the result in a TextView
        resultTextView.text = resultString.toString()
    }






    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tflite.close()
    }
}
