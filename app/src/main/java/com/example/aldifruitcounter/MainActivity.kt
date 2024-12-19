package com.example.aldifruitcounter

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.fruitcounter.R
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var captureButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        captureButton = findViewById(R.id.captureButton)
        resultTextView = findViewById(R.id.resultTextView)

        // Load TensorFlow Lite model
        val tfliteModel = ByteBuffer.wrap(assets.open("model.tflite").use { it.readBytes() })
        tflite = Interpreter(tfliteModel)

        // Initialize Camera
        cameraExecutor = Executors.newSingleThreadExecutor()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = androidx.camera.core.Preview.Builder().build()
                .also { it.setSurfaceProvider(viewFinder.surfaceProvider) }

            imageCapture = ImageCapture.Builder().build()

            cameraProvider.bindToLifecycle(
                this,
                androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageCapture
            )
        }, ContextCompat.getMainExecutor(this))

        captureButton.setOnClickListener { captureAndAnalyzeImage() }
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
        return ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri))
    }

    private fun analyzeImage(bitmap: Bitmap) {
        // Resize bitmap to match model input size
        val inputImage = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // Convert to ByteBuffer
        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        inputBuffer.rewind()
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = inputImage.getPixel(x, y)
                inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f) // Red
                inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255.0f)  // Green
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)       // Blue
            }
        }

        // Run inference
        val outputBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, 10), DataType.FLOAT32) // Adjust output size
        tflite.run(inputBuffer, outputBuffer.buffer.rewind())

        // Display result
        val result = outputBuffer.floatArray
        resultTextView.text = "Prediction: ${result.indices.maxByOrNull { result[it] }}"
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tflite.close()
    }
}
