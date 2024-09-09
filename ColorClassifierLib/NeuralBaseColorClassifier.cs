using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace ColorClassifierLib
{
    public class NeuralBaseColorClassifier : IBaseColorClassifier
    {
        private readonly InferenceSession _inferenceSession;

        public NeuralBaseColorClassifier(string classifierModelPath)
        {
            if (classifierModelPath == null) throw new ArgumentNullException(nameof(classifierModelPath));
            _inferenceSession = new InferenceSession(classifierModelPath);

        }
        public IDictionary<BaseColor, float> GetBaseColorConfidences(byte r, byte g, byte b)
        {
            float[] inputVector = { r, g, b };

            var inputTensor = new DenseTensor<float>(inputVector,
                // one row, 3 values
                new[] { 1, 3 });

            // Create input container
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("dense_12_input", inputTensor)
            };

            // Run inference
            using var results = _inferenceSession.Run(inputs);

            var outputTensor = results[0].AsTensor<float>();

            int numSamples = outputTensor.Dimensions[0];

            //Batch size ?
            if (numSamples != 1)
                throw new InvalidOperationException($"Expected 1 got {numSamples}");

            // Color classes
            int numClasses = outputTensor.Dimensions[1];

            var colorConfidenceMap = new Dictionary<BaseColor, float>();

            // Find the index of the highest confidence score
            float maxConfidence = float.MinValue;

            for (int j = 0; j < numClasses; j++)
            {
                var confidence = outputTensor[0, j];
                var objColor = (BaseColor)j;

                if (confidence > maxConfidence)
                {
                    maxConfidence = outputTensor[0, j];
                }

                colorConfidenceMap.Add(objColor, confidence);
            }

            return colorConfidenceMap;
        }
    }
}
