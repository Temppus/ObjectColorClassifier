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
        public BaseColor GetBaseColor(byte r, byte g, byte b)
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

            // Assuming output tensor shape is [unk__32, 11]
            int numSamples = outputTensor.Dimensions[0];
            int numClasses = outputTensor.Dimensions[1];

            var colorConfidenceMap = new Dictionary<BaseColor, float>();

            for (int i = 0; i < numSamples; i++)
            {
                // Find the index of the highest confidence score
                float maxConfidence = float.MinValue;
                int colorIndex = -1;
                for (int j = 0; j < numClasses; j++)
                {
                    var confidence = outputTensor[i, j];
                    var objColor = (BaseColor)j;

                    if (confidence > maxConfidence)
                    {
                        maxConfidence = outputTensor[i, j];
                        colorIndex = j;
                    }

                    colorConfidenceMap.Add(objColor, confidence);
                }

                return colorConfidenceMap.MaxBy(x => x.Value).Key;
            }

            throw new InvalidOperationException("Sanity");
        }
    }
}
